### GAViKO: 3D ViT fine-tuning and prompting for medical image classification

GAViKO is a PyTorch-based framework for 3D medical image classification built on top of Vision Transformers (ViT). It supports multiple parameter-efficient fine-tuning (PEFT) and prompting strategies for 3D inputs (e.g., volumetric MRI saved as `.npz`).

![![alt text](GAViKO Architecture)](src/asset/image.png)



Supported methods (select via `--method`):
- `gaviko`: custom global-local prompt fusion (default)
- `linear`: linear-probe head on frozen ViT
- `bitfit`: train only biases and head
- `adaptformer`: Adapter-based fine-tuning
- `dvpt`: Dynamic Visual Prompt Tuning
- `evp`: Explicit Visual Prompting (frequency/features-guided)
- `ssf`: Scaling & Shifting Features (SSF)
- `melo`: LoRA for ViT (MeLO)
- `deep_vpt` / `shallow_vpt`: Visual Prompt Tuning variants
- `fft`: ViT with FFT-related configuration (used by `evp`)


### References for supported methods

- AdaptFormer: [GitHub repository](https://github.com/ShoufaChen/AdaptFormer) • [Paper](https://arxiv.org/abs/2205.13535)
- DVPT (Dynamic Visual Prompt Tuning): [GitHub repository](https://github.com/NKUhealong/DVPT) • [Paper](https://arxiv.org/abs/2307.09787)
- VPT (deep_vpt/shallow_vpt): [GitHub repository](https://github.com/KMnP/vpt) • [Paper](https://arxiv.org/abs/2203.12119)
- EVP (Explicit Visual Prompting): [GitHub repository](https://github.com/NiFangBaAGe/Explicit-Visual-Prompt) • [Paper](https://arxiv.org/abs/2305.18476)
- SSF (Scaling & Shifting Features): [GitHub repository](https://github.com/dongzelian/SSF) • [Paper](https://arxiv.org/abs/2210.08823)
- MeLO (LoRA-ViT for ViT): [GitHub repository](https://github.com/JamesQFreeman/LoRA-ViT) • [Paper](https://arxiv.org/abs/2311.08236)
- ViT 3D backbone adaptation: [GitHub repository](https://github.com/lucidrains/vit-pytorch) • [Paper](https://arxiv.org/abs/2311.08236)
- Focal Loss implementation and paper: [GitHub repository](https://github.com/mathiaszinnen/focal_loss_torch) • [Paper](https://arxiv.org/abs/1708.02002)


### Repository structure

```
src/
  configs/               # Example configurations
  data/
    dataset.py           # Dataset and inference-only dataset
  losses/
    focal_loss.py
  model/
    vision_transformer.py
    gaviko.py            # Main GAViKO model
    adaptformer.py
    dvpt.py
    evp.py
    melo.py
    ssf.py
    vpt.py
  utils/
    load_pretrained.py   # Download/convert ViT weights via timm
    logging.py           # CSV logger, memory/flops utilities
  train.py               # Training entrypoint
  eval.py                # Evaluation on labeled split, writes metrics + CSV
  inference.py           # Folder inference without labels
requirements.txt
README.md
```


### Environment and installation

1) Create environment (Python 3.10+ recommended)

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
```

2) Install PyTorch matching your CUDA (or CPU)

```bash
# Example (CUDA 12.1); pick the right index-url from pytorch.org
pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio
```

3) Install project dependencies

```bash
pip install -r requirements.txt

# The code also uses the following packages which may not be pinned in requirements:
pip install timm colorama torchprofile deepspeed  # deepspeed is optional
```

Notes
- The first run will download ImageNet-pretrained ViT weights via `timm` and cache them under `./pretrained/`.
- If you hit a version conflict (e.g., PyTorch version), prefer the PyTorch you installed in step 2.


### Data format

The framework expects 3D volumes stored as `.npz` files containing a `data` array shaped like `(frames, height, width)`, e.g. `(120, 160, 160)`. During loading, a channel dim is added to form `(1, D, H, W)`.

For training/evaluation, provide a CSV with at least the following columns:
- `mri_path`: path (absolute or relative) to the `.npz` file
- `kl_grade`: integer class label
- `subset`: one of `train`, `val`, `test`

Optionally, you can set `data.image_folder` in configs; when set, paths in the CSV can be basenames and will be joined with this folder.

Example CSV rows:

```csv
mri_path,kl_grade,subset
patient_0001.npz,2,train
patient_0002.npz,3,val
patient_0003.npz,1,test
```


### Configuration

Configs live in `src/configs/`. Despite the `.yaml` extension, they use YAML/JSON-like flow mappings. Key groups:
- `utils`: logging, phase, output dirs
- `data`: batch size, workers, CSV path, images folder
- `model`: ViT backbone (`vit-b16`, `vit-t16`, `vit-s16`, `vit-l16`), image/patch sizes, prompts/adapter settings, number of classes
- `train`: epochs, lr, scheduler, early stopping, mixed precision, DeepSpeed switches
- `wandb`: Weights & Biases logging (optional)

You can switch the method at runtime via `--method`. For VPT specifically:
- `--method deep_vpt` sets `deep_prompt=True`
- `--method shallow_vpt` sets `deep_prompt=False`

Backbones are mapped internally and downloaded via `timm`.


### Quick start

Run from the repository root.

Train (example: GAViKO method)

```bash
python src/train.py \
  --config src/configs/gaviko.yaml \
  --method gaviko \
  --results_dir ./outputs
```

Evaluate on labeled split (writes metrics and a CSV of predictions)

```bash
python src/eval.py \
  --config src/configs/gaviko.yaml \
  --method gaviko \
  --checkpoint path/to/weights.pt \
  --results_dir ./outputs
```

Folder inference without labels (predict all `.npz` in a folder)

```bash
python src/inference.py \
  --config src/configs/gaviko.yaml \
  --method gaviko \
  --checkpoint path/to/weights.pt \
  --results_dir ./outputs
```

Notes
- Checkpoint files saved during training include only trainable parameters (e.g., prompts, adapters, head). At inference, the loader merges them with the vanilla ViT weights from `timm`.
- Outputs are written to `--results_dir` with versioned filenames. Training logs (CSV + .log) are under `./src/log/` by default.


### Choosing a method

You can substitute `--method` in the commands above with any of:
- `gaviko`, `linear`, `bitfit`, `adaptformer`, `dvpt`, `evp`, `ssf`, `melo`, `deep_vpt`, `shallow_vpt`

Make sure the chosen config aligns with the method (e.g., prompt counts for VPT/DVPT, EVP’s FFT options, LoRA ranks for MeLO). The provided files in `src/configs/` are good starting points and can be adapted.


### Checkpoints and logging

- Weights: `train.save_dir/experiments/{method}/{method}_{backbone}_best_model_epoch{E}_acc{ACC}.pt`
- CSV logs: versioned files under `utils.log_dir` with fields for per-step and per-epoch metrics
- Python logs: timestamped files under `utils.log_dir`
- FLOPs/memory: optional utilities in `utils/logging.py`


### DeepSpeed (optional)

DeepSpeed support is wired but disabled by default (`train.deepspeed.enabled: False`). To use it:
1) Install deepspeed: `pip install deepspeed`
2) Set `train.deepspeed.enabled: True` and point `train.deepspeed.config` to a valid config JSON. The example file `src/configs/deepspeed.yaml` is a template; adjust or replace with a proper DeepSpeed JSON.


### Troubleshooting

- If imports fail (e.g., `from data.dataset import ...`), ensure you run the scripts from the repository root using `python src/<script>.py` so that `src/` is on `sys.path`.
- If `timm` cannot download weights (no internet), pre-download ViT weights or cache them and place under `./pretrained/` with the expected filenames.
- If you see mixed precision or CUDA errors, set `model.fp16: False` and/or `train.fp16: False` in the config to fall back to FP32.
- If you run on CPU, use smaller batch sizes and consider `model.freeze_vit=True` methods.


### License and acknowledgments

This repository adapts and builds upon several open-source implementations referenced in the model files. Please see inline headers in `src/model/*.py` for upstream links and licenses.


