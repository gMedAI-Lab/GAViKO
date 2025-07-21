import os
import csv
from datetime import datetime
import torch
import logging
import psutil
import sys
import numpy as np
class CSVLogger:
    def __init__(self, log_dir, filename_prefix='train_log', fields=['epoch', 'loss', 'acc', 'lr']):
        """
        Initialize CSV logger.
        
        Args:
            log_dir (str): Directory where the log files will be saved.
            filename_prefix (str): Prefix for the log file name.
            fields (list): List of field names to log (e.g. ['epoch', 'loss', 'acc', 'lr']).
        """
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)

        # Generate timestamped filename
        # timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        # use ver increment if file already exists
        version = 1
        while True:
            filename = f"{filename_prefix}_v{version}.csv"
            self.filename = os.path.join(log_dir, filename)
            if not os.path.exists(self.filename):
                break
            version += 1

        self.fields = fields

        # Create file and write header if not exists
        if not os.path.exists(self.filename):
            with open(self.filename, mode='w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self.fields)
                writer.writeheader()

    def log(self, data):
        """
        Log a row of data to the CSV file.
        
        Args:
            data (dict): Dictionary with keys matching the `fields`.
        """
        with open(self.filename, mode='a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.fields)
            writer.writerow(data)

    def get_file_path(self):
        """Return the path to the current log file."""
        return self.filename
    
def setup_logging(log_level=logging.INFO, log_dir='./src/log'):
    """Configure logging to file and console"""
    # Create log directory if it doesn't exist
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        
    # Log file with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f"{timestamp}.log")
    
    # Set up logging
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    logging.info(f"Logging initialized. Log file: {log_file}")

def analyze_model_computation(parameters, flops, verbose=True):
    """
    Simple model analysis focusing on memory requirements and basic stats.
    
    Args:
        parameters (int): Number of model parameters
        flops (int, optional): Number of FLOPs per forward pass
        verbose (bool): Whether to print analysis
    
    Returns:
        dict: Dictionary containing computed metrics
    """
    
    # Memory calculations (in bytes)
    memory_fp32 = parameters * 4
    memory_fp16 = parameters * 2
    memory_int8 = parameters * 1
    
    # Convert to MB
    memory_fp32_mb = memory_fp32 / (1024 * 1024)
    memory_fp16_mb = memory_fp16 / (1024 * 1024)
    memory_int8_mb = memory_int8 / (1024 * 1024)
    
    # Training memory estimate (includes gradients + optimizer states)
    training_memory_gb = memory_fp32_mb * 3 / 1024
    
    results = {
        'parameters': parameters,
        'parameters_millions': round(parameters / 1e6, 2),
        'memory_mb': {
            'fp32': round(memory_fp32_mb, 1),
            'fp16': round(memory_fp16_mb, 1),
            'int8': round(memory_int8_mb, 1)
        },
        'training_memory_gb': round(training_memory_gb, 1)
    }
    
    # Add FLOP info if provided
    if flops is not None:
        gflops = flops / 1e9
        tflops = flops / 1e12
        results.update({
            'flops': flops,
            'gflops': round(gflops, 2),
            'tflops': round(tflops, 3)
        })
    
    if verbose:
        print("=" * 40)
        print("MODEL ANALYSIS")
        print("=" * 40)
        
        print(f"Parameters: {parameters:,} ({results['parameters_millions']}M)")
        
        if flops is not None:
            print(f"FLOPs: {flops:,}")
            print(f"Compute: {results['gflops']} GFLOP ({results['tflops']} TFLOP)")
        
        print(f"\nMemory Requirements:")
        print(f"  FP32: {results['memory_mb']['fp32']} MB")
        print(f"  FP16: {results['memory_mb']['fp16']} MB")
        print(f"  INT8: {results['memory_mb']['int8']} MB")
        print(f"  Training: ~{results['training_memory_gb']} GB")
    
    return results
from colorama import Fore, Style, init
import torch

# Initialize colorama for cross-platform support
class MemoryUsageLogger:
    def __init__(self, verbose=True):
        init(autoreset=True)
        self.verbose = verbose
        self.index = 0
    def display_ram_usage(self, model=None, data=None, device=None):
        """
        Calculates and displays GPU memory usage for loading a model and data with colorful output.

        Args:
            model (nn.Module): A PyTorch model (optional).
            data (torch.Tensor or list): Input data - single tensor or list of tensors (optional).
            device (str or torch.device): Device to use ('cuda', 'cuda:0', etc.). If None, auto-detects.
            verbose (bool): Whether to print detailed memory information.

        Returns:
            dict: Dictionary containing memory usage statistics in MB.
        """
        # Auto-detect device if not specified
        if not self.index == 0:
            return None
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            device = torch.device(device)

        # Check if CUDA is available and device is CUDA
        if not torch.cuda.is_available():
            if self.verbose:
                print(Fore.RED + "CUDA is not available. Cannot measure GPU memory usage.")
            return None
        
        if device.type != 'cuda':
            if self.verbose:
                print(Fore.RED + f"Device {device} is not a CUDA device. Cannot measure GPU memory usage.")
            return None

        # Get GPU device index
        gpu_index = device.index if device.index is not None else 0

        # Reset peak memory stats for the specific GPU
        torch.cuda.reset_peak_memory_stats(gpu_index)

        # Initial memory stats
        initial_allocated = torch.cuda.memory_allocated(gpu_index) / (1024 ** 2)  # MB
        initial_reserved = torch.cuda.memory_reserved(gpu_index) / (1024 ** 2)     # MB
        total_memory = torch.cuda.get_device_properties(gpu_index).total_memory / (1024 ** 2)  # MB

        if self.verbose:
            print(Fore.CYAN + f"\nGPU {gpu_index} ({torch.cuda.get_device_name(gpu_index)})")
            print(Fore.GREEN + f"Total GPU memory: {total_memory:.2f} MB")
            print(Fore.YELLOW + f"Initial GPU memory allocated: {initial_allocated:.2f} MB")
            print(Fore.YELLOW + f"Initial GPU memory reserved: {initial_reserved:.2f} MB")

        model_size = 0
        # Load model to GPU
        if model is not None:
            # Check if model is already on the correct device
            if next(model.parameters(), None) is None:
                if self.verbose:
                    print(Fore.RED + "Warning: Model has no parameters")
            elif next(model.parameters()).device != device:
                model.to(device)
            
            # Calculate model size more accurately
            model_size = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 ** 2)  # MB
            if self.verbose:
                print(Fore.MAGENTA + f"Model size: {model_size:.2f} MB")

        data_size = 0
        # Load data to GPU
        if data is not None:
            if isinstance(data, (list, tuple)):
                # Handle multiple tensors
                data_moved = []
                total_elements = 0
                total_bytes = 0
                for tensor in data:
                    if isinstance(tensor, torch.Tensor):
                        tensor_moved = tensor.to(device)
                        data_moved.append(tensor_moved)
                        total_elements += tensor.numel()
                        total_bytes += tensor.numel() * tensor.element_size()
                data = data_moved
                data_size = total_bytes / (1024 ** 2)  # MB
            elif isinstance(data, torch.Tensor):
                # Handle single tensor
                data = data.to(device)
                data_size = data.numel() * data.element_size() / (1024 ** 2)  # MB
            else:
                if self.verbose:
                    print(Fore.RED + "Warning: Data is not a torch.Tensor or list of tensors")
            
            if self.verbose and data_size > 0:
                print(Fore.BLUE + f"Data size: {data_size:.2f} MB")

        # Final memory stats
        final_allocated = torch.cuda.memory_allocated(gpu_index) / (1024 ** 2)  # MB
        final_reserved = torch.cuda.memory_reserved(gpu_index) / (1024 ** 2)    # MB

        # Peak memory stats
        peak_allocated = torch.cuda.max_memory_allocated(gpu_index) / (1024 ** 2)  # MB
        peak_reserved = torch.cuda.max_memory_reserved(gpu_index) / (1024 ** 2)    # MB

        # Calculate actual usage
        memory_increase = final_allocated - initial_allocated
        available_memory = total_memory - final_allocated

        if self.verbose:
            print(Fore.CYAN + "\n--- Final Memory Stats ---")
            print(Fore.GREEN + f"Final GPU memory allocated: {final_allocated:.2f} MB")
            print(Fore.GREEN + f"Final GPU memory reserved: {final_reserved:.2f} MB")
            print(Fore.YELLOW + f"Peak GPU memory allocated: {peak_allocated:.2f} MB")
            print(Fore.YELLOW + f"Peak GPU memory reserved: {peak_reserved:.2f} MB")
            print(Fore.MAGENTA + f"Memory increase: {memory_increase:.2f} MB")
            print(Fore.BLUE + f"Available memory: {available_memory:.2f} MB")
            print(Fore.WHITE + f"Memory utilization: {(final_allocated / total_memory) * 100:.2f}%")

        return {
            'gpu_index': gpu_index,
            'gpu_name': torch.cuda.get_device_name(gpu_index),
            'total_memory': total_memory,
            'initial_allocated': initial_allocated,
            'initial_reserved': initial_reserved,
            'final_allocated': final_allocated,
            'final_reserved': final_reserved,
            'peak_allocated': peak_allocated,
            'peak_reserved': peak_reserved,
            'memory_increase': memory_increase,
            'available_memory': available_memory,
            'utilization_percent': (final_allocated / total_memory) * 100,
            'model_size': model_size,
            'data_size': data_size
        }
    def display_before_forward_pass(self, model=None, data=None, device=None):
        """
        Display memory usage before the forward pass.
        
        Args:
            model (nn.Module): A PyTorch model (optional).
            data (torch.Tensor or list): Input data - single tensor or list of tensors (optional).
            device (str or torch.device): Device to use ('cuda', 'cuda:0', etc.). If None, auto-detects.
        """
        if self.index == 0:
            print(Fore.CYAN + f"\nBefore forward pass:")
            self.display_ram_usage(model=model, data=data, device=device)

    def display_after_forward_pass(self, model=None, data=None, device=None):
        """
        Display memory usage after the forward pass.
        
        Args:
            model (nn.Module): A PyTorch model (optional).
            data (torch.Tensor or list): Input data - single tensor or list of tensors (optional).
            device (str or torch.device): Device to use ('cuda', 'cuda:0', etc.). If None, auto-detects.
        """
        if self.index == 0:
            print(Fore.CYAN + f"\nAfter forward pass:")
            self.display_ram_usage(model=model, data=data, device=device)
    def display_after_moving_data_to_gpu(self, data=None, device=None):
        """
        Display memory usage after moving data to GPU.
        
        Args:
            data (torch.Tensor or list): Input data - single tensor or list of tensors (optional).
            device (str or torch.device): Device to use ('cuda', 'cuda:0', etc.). If None, auto-detects.
        """
        if self.index == 0:
            print(Fore.CYAN + f"\nAfter moving data to GPU:")
            self.display_ram_usage(data=data, device=device)
    def display_after_backward_pass(self, model=None, data=None, device=None):
        """
        Display memory usage after the backward pass.
        
        Args:
            model (nn.Module): A PyTorch model (optional).
            data (torch.Tensor or list): Input data - single tensor or list of tensors (optional).
            device (str or torch.device): Device to use ('cuda', 'cuda:0', etc.). If None, auto-detects.
        """
        if self.index == 0:
            print(Fore.CYAN + f"\nAfter backward pass:")
            self.display_ram_usage(model=model, data=data, device=device)

    def display_after_optimization_step(self, model=None, data=None, device=None):
        """
        Display memory usage after the optimization step.
        
        Args:
            model (nn.Module): A PyTorch model (optional).
            data (torch.Tensor or list): Input data - single tensor or list of tensors (optional).
            device (str or torch.device): Device to use ('cuda', 'cuda:0', etc.). If None, auto-detects.
        """
        if self.index == 0:
            print(Fore.CYAN + f"\nAfter optimization step:")
            self.display_ram_usage(model=model, data=data, device=device)