#!/bin/bash
#SBATCH --job-name=gpu_check
#SBATCH --partition=ASL-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Ten-minute check of the GPU node, driver, and Conda environment before a long run.
# Submit from the project root after setup: sbatch slurm/gpu_check.sbatch
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
echo "Node: $(hostname) | Job: $SLURM_JOB_ID | $(date)"
# As in pgat-length: the login node's /opt/miniconda3 is not on the compute node,
# so put the environment on PATH instead of activating it.
ENV_DIR="${ENV_DIR:-$HOME/.conda/envs/${ENV_NAME:-swinunet}}"
test -x "$ENV_DIR/bin/python" || { echo "Conda environment not found: $ENV_DIR" >&2; exit 1; }
export PATH="$ENV_DIR/bin:$PATH"
export LD_LIBRARY_PATH="$ENV_DIR/lib:${LD_LIBRARY_PATH:-}"
export ENCODER="${ENCODER:-tu-hrnet_w32}"
export HF_HUB_OFFLINE=1  # as in the training job: the weights must come from the cache
nvidia-smi
python - <<'PY'
import torch, torchvision, timm
import segmentation_models_pytorch as smp
print('PyTorch:', torch.__version__, '| torchvision:', torchvision.__version__, '| CUDA build:', torch.version.cuda)
print('segmentation_models_pytorch:', smp.__version__, '| timm:', timm.__version__)
assert torch.cuda.is_available(), 'CUDA unavailable: check the GPU driver and PyTorch wheel compatibility'
memory = torch.cuda.get_device_properties(0).total_memory / 2 ** 30
print(f'GPU: {torch.cuda.get_device_name(0)} | {memory:.0f} GB')
x = torch.randn(4096, 4096, device='cuda')
print('CUDA matmul check:', tuple((x @ x).shape))
PY
python scripts/encoder_weights.py "$ENCODER" || {
    echo "MISSING - run on the login node: ENCODER=$ENCODER bash scripts/setup_conda.sh" >&2; exit 1; }
# The network itself, in half precision as in training: one training step on a batch of the
# default size and crop (12 x 256 px, every output and the whole loss), then the largest
# evaluation tile.
python - <<'PY'
import os, sys
sys.path.insert(0, 'src')
import torch
from hrcracknet.models import HRCrackNet
from hrcracknet.losses import HairlineCrackLoss
model = HRCrackNet(os.environ["ENCODER"], pretrained=True).cuda().to(memory_format=torch.channels_last).train()
images = torch.rand(12, 3, 256, 256, device='cuda').contiguous(memory_format=torch.channels_last)
masks = (torch.rand(12, 1, 256, 256, device='cuda') > 0.97).float()
with torch.autocast('cuda', dtype=torch.float16):
    outputs = model(images)
loss = HairlineCrackLoss()(outputs, masks, torch.ones_like(masks))
loss.backward()
peak = torch.cuda.max_memory_allocated() / 2 ** 30
print(f'HRCrackNet on {os.environ["ENCODER"]}: training step on 12 x 256 x 256, loss {loss.item():.3f}, '
      f'{peak:.1f} GB')
model.eval()
torch.cuda.reset_peak_memory_stats()
with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16):
    logits = model(torch.rand(1, 3, 1024, 1024, device='cuda').contiguous(memory_format=torch.channels_last))
peak = torch.cuda.max_memory_allocated() / 2 ** 30
print(f'HRCrackNet on {os.environ["ENCODER"]}: output {tuple(logits.shape)}, finite {bool(torch.isfinite(logits).all())}, '
      f'{peak:.1f} GB for one 1024 x 1024 tile')
PY
echo "GPU check passed."
