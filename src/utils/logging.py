import os
import csv
from datetime import datetime
import torch

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