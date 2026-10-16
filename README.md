# HRCrackNet: hairline crack segmentation

A network built for cracks one or two pixels wide whose contrast is close to
that of the concrete texture: an HRNet-W32 encoder that keeps a 1/4-scale
branch through the whole network, directional strip convolutions that add up
the evidence along a crack, a full-resolution hairline stem, and a
distance-field output that gives the network a graded target around thin,
hand-drawn labels.

The project is organised and run like the
[Swin U-Net](https://github.com/poudelavik1/swinunet),
[FiLM U-Net](https://github.com/poudelavik1/filmunet),
[multi-decoder U-Net](https://github.com/poudelavik1/addunet) and
[UNet++](https://github.com/poudelavik1/Unet-) projects and shares their data
pipeline, augmentation library, loss, metrics and figures, so the runs can be
compared directly. Only the network is new.

```text
configs/             Training defaults and the GSD table
src/hrcracknet/      Augmentation, data, model, loss, evaluation, plotting, training modules
scripts/             Conda setup, encoder weights, training CLI, submission, ablation study, epoch monitoring
slurm/               ASL GPU check and training jobs
tests/               Workflow, trainer, model and study checks
docs/                Architecture and server guide
```

- [Architecture: why each part, and what the UNet++ runs showed](docs/ARCHITECTURE.md)
- [ASL setup, submission, the ablation study, monitoring and resume](docs/ASL_SETUP.md)
- [Training defaults](configs/training.toml) and [GSD table](configs/gsd.csv)

## The network in short

```text
image ──┬─ HRNet-W32 ── branches at 1/4, 1/8, 1/16, 1/32 (+ stem at 1/2)
        │       └─ all brought to 1/4, joined ─▶ strip context x2 ─▶ up to 1/2 (+ stem) ─▶ up to 1/1 ─▶ crack
        └─ hairline stem: RGB + black-hat / top-hat at full resolution ───────────────────────┘
   training only: crack outputs at 1/4 and 1/2 (deep supervision), distance field at 1/1
```

| Part | Switch | What it is for |
| --- | --- | --- |
| HRNet-W32 encoder | `--encoder` | Keeps 1/4-scale detail through every stage instead of recovering it from 1/32 |
| Strip context | `--no-strip-context` | Sums along horizontal, vertical and diagonal lines (faint cracks); the strip-pooling gate tells wandering cracks from straight formwork joints |
| Hairline stem | `--no-hairline-stem`, `--no-line-prior` | Full-resolution skip with a thin-line prior, as in the other trainers |
| Deep supervision | `--no-deep-supervision` | Crack outputs at 1/4 and 1/2 scale in training |
| Distance-field head | `--no-distance-head` | Graded target up to 5 px around each crack, forgiving labels a pixel or two off |

29.9 M parameters (UNet++ on ConvNeXt-Tiny: 35.1 M). The training settings,
augmentation (`all` preset), loss (BCE + Tversky + clDice), thresholds per
source and outputs are those of the UNet++ project.

## Ablation study

`scripts/submit_ablation.sh` trains eight runs, one after another: the full
network, one run without each part, the full network on the ConvNeXt-Tiny
encoder of the UNet++ runs, and a baseline without any of the parts.
`scripts/compare_runs.py` then draws them in one set of figures: test metrics
against the full network, tolerant F1 per source, validation curves, and
precision against recall. `STUDY=losses` trains the optimiser and loss grid of
the UNet++ project instead
([the study](docs/ARCHITECTURE.md#ablation-study)).

## Run locally

```bash
python -m pip install -r requirements.txt
```

```bash
python scripts/train.py --self-test
```

```bash
python scripts/train.py "E:\STRUCTURAL ENGINEERING\thesis\crackdetection\DATASETS\split_80_10_10" --output runs/local --gsd-csv configs/gsd.csv
```

`python scripts/train.py --help` lists every option, including one
`--<name>-p` probability per augmentation.

## First time on the server

```bash
cd ~
[ -d hrcracknet ] || git clone https://github.com/poudelavik1/HRCrackNet.git hrcracknet
cd hrcracknet
bash scripts/setup_conda.sh
```

Setup reuses the `swinunet` Conda environment, which already holds
`segmentation-models-pytorch` and `timm` from the UNet++ project, and caches
the pretrained HRNet-W32 and ConvNeXt-Tiny encoders, which the compute nodes
cannot download.

## Train on the server

```bash
cd ~/hrcracknet
git pull --ff-only
bash scripts/submit.sh "$HOME/dataset_20261011/split_80_10_10"
```

The whole ablation study, in the background on the login node:

```bash
cd ~/hrcracknet && nohup bash scripts/submit_ablation.sh "$HOME/dataset_20261011/split_80_10_10" > logs/ablation.log 2>&1 &
```

All variables are listed in the [server guide](docs/ASL_SETUP.md).

Run the tests with `python -m unittest discover -s tests -v`.
