# HRCrackNet hairline crack training on ASL

## Project layout

```text
configs/training.toml         Core training defaults
configs/gsd.csv               GSD (mm per pixel) of each source
src/hrcracknet/               Augmentation, data, models, losses, evaluation, plots, training
scripts/                      Conda setup, encoder weights, submission, training CLI, epoch monitor,
                              ablation study, comparison figures
slurm/train_hrcracknet.sbatch ASL CUDA training job
slurm/gpu_check.sbatch        Ten-minute GPU, environment and network check
docs/ARCHITECTURE.md          Encoder, strip context, decoder, heads and the ablation study
tests/                        Workflow, trainer, model and study checks
```

The workflow is that of the Swin U-Net, FiLM U-Net, multi-decoder U-Net and
UNet++ projects.

## Upload to GitHub from Windows

Repository: https://github.com/poudelavik1/HRCrackNet
For future local updates, run from this project directory:

```powershell
git add .gitignore .gitattributes README.md requirements.txt environment.yml pyproject.toml configs src scripts slurm docs tests
git commit -m "Update training code"
git push origin main
```

Do not commit the dataset, ZIP, checkpoints, or credentials. A private repository
requires GitHub authentication when cloning (SSH key or credential helper).

## Server setup (once)

Connect to the ASL login node as for the other projects. The project is cloned
into `~/hrcracknet`, the folder name used below.

```bash
cd ~
[ -d hrcracknet ] || git clone https://github.com/poudelavik1/HRCrackNet.git hrcracknet
cd hrcracknet
bash scripts/setup_conda.sh
```

Setup uses the `swinunet` Conda environment of the Swin U-Net project (Python
3.11, PyTorch 2.9.1 / torchvision 0.24.1 CUDA 12.6 wheels) and creates it if it
does not exist. `segmentation-models-pytorch` 0.5.0 and `timm` are already in
it from the UNet++ project; setup checks them and installs this project next to
the others. Set `ENV_NAME` to use another environment, and set the same
`ENV_NAME` (or `ENV_DIR`) when submitting. Jobs do not call `conda activate`,
because `/opt/miniconda3` exists only on the login node; they put
`~/.conda/envs/$ENV_NAME/bin` on `PATH` directly.

Setup also downloads the pretrained HRNet-W32 encoder and the ConvNeXt-Tiny
encoder of the ablation run into `~/.cache/huggingface`, so compute nodes need
no internet. Another encoder needs its weights cached from the login node
before it is used:

```bash
ENCODER=tu-hrnet_w18 bash scripts/setup_conda.sh
```

Jobs run with `HF_HUB_OFFLINE=1` and read the weights from that cache. When the
pretrained HRNet is loaded, timm lists the keys of its classification head as
"Unexpected keys ... found while loading pretrained weights". That is expected:
the network uses the four branch outputs, not the classification head; the
tests check that every encoder weight is loaded.

## Check the GPU (recommended before the first run)

```bash
sbatch slurm/gpu_check.sbatch
squeue -u "$USER"
cat logs/gpu_check_JOBID.out logs/gpu_check_JOBID.err
```

The output shows the GPU, driver, PyTorch CUDA build and package versions,
loads the cached encoder weights, runs one training step on a batch of the
default size (12 crops of 256 px, every output and the whole loss) and the
network on a 1024 x 1024 evaluation tile in half precision, and prints the GPU
memory each took. It ends with `GPU check passed.` Check the ConvNeXt variant
with `ENCODER=tu-convnext_tiny sbatch --export=ALL slurm/gpu_check.sbatch`.

## Submit CUDA training

The dataset must contain `train/images`, `train/masks`, `val/images`,
`val/masks`, `test/images`, and `test/masks`.

```bash
bash scripts/submit.sh "$HOME/dataset_20261011/split_80_10_10"
```

That trains the full network: HRCrackNet on HRNet-W32 with every augmentation
on (`all` preset). Core defaults live in `configs/training.toml`. CLI flags take
precedence. Direct training on an allocated compute node:

```bash
python scripts/train.py "$HOME/dataset_20261011/split_80_10_10" --output runs/manual --gsd-csv configs/gsd.csv
```

Default request: ASL-gpu, one GPU, four CPU cores, 32 GB host RAM, two days.
Request a shorter job with `SBATCH_TIMELIMIT=1-00:00:00` before the command.
Submission prints the job ID, result directory, and monitoring commands.
Each new job uses a separate output directory and does not resume old weights.
On ASL a second job of the same user has been removed from the queue while the first was
running, so submit the next run after the last one ends.

Variables in front of the command change the run:

| Variable | Effect | Default |
| --- | --- | --- |
| `EPOCHS`, `BATCH_SIZE`, `WORKERS` | Core settings | `configs/training.toml` (80, 12, 4) |
| `ENCODER` | Any smp encoder: `tu-hrnet_w32`, `tu-hrnet_w18`, `tu-hrnet_w48`, `tu-convnext_tiny`, ... | `tu-hrnet_w32` |
| `AUG_PRESET` | `all`, `heavy`, `medium`, `light`, `reference`, `none` | `all` |
| `OPTIMIZER` | `adamw` or `adam` | `adamw` |
| `LOSS` | Loss terms, separated by spaces: `bce`, `focal`, `dice`, `tversky`, `cldice`, or `hairline` (BCE + Tversky + clDice) | `hairline` |
| `TRAIN_ARGS` | Further trainer options, in one quoted string | none |
| `GSD_CSV` | GSD table (data pipeline and reports only) | `configs/gsd.csv` |
| `SELECT_PATTERN`, `TARGET_WEIGHT` | Select the model and threshold on one source | all sources |
| `INIT_WEIGHTS`, `LEARNING_RATE` | Fine-tune an earlier `best.pt` of this trainer | ImageNet encoder |
| `LABEL_TOLERANCE` | Pixels around each label left out of the loss | 0 |
| `SAVE_PREDICTIONS` | Test overlays to save | 40 |
| `EVALUATE_ONLY` | `1`: score the `best.pt` of an earlier run again, no training (below) | `0` |

`TRAIN_ARGS` passes any option of `python scripts/train.py --help` to the job,
for example a network without the strip context and the distance head:

```bash
TRAIN_ARGS="--no-strip-context --no-distance-head" bash scripts/submit.sh "$HOME/dataset_20261011/split_80_10_10"
```

The network options are `--no-strip-context`, `--no-distance-head`,
`--no-hairline-stem`, `--no-line-prior`, `--no-deep-supervision`, and the sizes
`--context-channels`, `--strip-blocks`, `--strip-length`, `--pool-length`,
`--decoder-channels`, `--distance-weight`, `--distance-margin`.

To compare with the Swin and FiLM U-Net runs, which used the reference
augmentation, add `AUG_PRESET=reference`.

By default the best checkpoint and the threshold are chosen on the validation
images of every source pooled by pixel, which the thick-crack sources dominate.
Every source family also gets its own threshold, calibrated on its validation
images, and the test metrics are reported at both (see
[the thresholds](ARCHITECTURE.md#thresholds)). `SELECT_PATTERN` (a regex on
validation file names) chooses them on a target source instead and samples
that source's training images `TARGET_WEIGHT` times as often (default 1.5). For
the row/column tiles:

```bash
SELECT_PATTERN='_r[0-9]+_c[0-9]+' bash scripts/submit.sh "$HOME/dataset_20261011/split_80_10_10"
```

Only use `SLURM_ACCOUNT` / `SLURM_PARTITION` values confirmed by the
administrator. A pending job can show `InvalidAccount` on this server and still
start on its own. Pending jobs produce no epoch metrics until they start.

The UNet++ run (12,736 training tiles, batch 12) took 210 s per epoch on an RTX
A6000, 4.7 hours for 80 epochs. On a CPU one HRCrackNet training step took 6.0 s
against 7.5 s for UNet++, so expect a similar time; the first epochs in the log
show it. Reduce the batch size if GPU memory is exhausted (`BATCH_SIZE=8`); if
evaluation exhausts memory, reduce the tile (`TRAIN_ARGS="--eval-tile 768"`).
Early stopping defaults to 15 epochs without improvement, so training can stop
before the requested epoch count.

## The GSD table

`configs/gsd.csv` gives the GSD of each source in mm per pixel:

```text
pattern,gsd_mm_per_px
^EH104_,0.6511
^RAMA8_EAST-,2.5
```

`pattern` is a regular expression searched in the image file name. The first
matching row gives the GSD, and a file without a match (CRACK500, the photos)
has an unknown GSD. Write `[0-9]` rather than `\d`: the submission script
checks the patterns with `grep -E` and the trainer applies them as Python
regular expressions.

The network does not use the GSD. The table serves the data pipeline: the
mixing augmentations (mosaic, CutMix, MixUp, copy-paste) only combine tiles of
one source and one GSD, and the test metrics are reported per GSD value.
Submission prints how many training images each pattern matches.
`GSD_CSV=/absolute/path.csv` uses another table.

## Monitor performance

Use the actual job ID and output path printed by the submission script:

```bash
squeue -u "$USER"
tail -F logs/hrcracknet_JOBID.out
python scripts/monitor_epochs.py /absolute/path/to/run --watch
cat logs/hrcracknet_JOBID.err
```

Ctrl+C stops the viewer; the submitted job keeps running after SSH disconnects.
The monitor prints a row only after training AND validation finish an epoch.
All epoch values remain in `training_history.csv`, and `validation_by_source.csv`
holds the same metrics for every source separately (the log prints the three
weakest after each epoch).

At the start the log prints the source table, the number of enabled
augmentations and the network with its parts. After training, the run
directory holds:

- `best.pt`, resumable `last.pt`, `configuration.json`, `training_history.csv`
  (with `train_distance_loss`), `validation_by_source.csv`,
  `val_threshold_sweep.csv`
- `test_metrics.json`: pooled, per source and per GSD value at the shared
  threshold, and under `with_source_thresholds` with every source at its own
- `test_predictions/`: for each saved test image the probability map and the
  overlay (yellow = hit, red = false positive, green = missed)
- `augmentation_counts.json`, `contrast_coverage.json`, `plots/`, and the Git
  commit and package versions of the job

## Ablation study

One command trains the eight variants of the
[ablation study](ARCHITECTURE.md#ablation-study), one job after another,
because ASL has removed a second job of the same user from the queue. It runs
on the login node in the background and keeps going after you log out:

```bash
cd ~/hrcracknet && nohup bash scripts/submit_ablation.sh "$HOME/dataset_20261011/split_80_10_10" > logs/ablation.log 2>&1 &
```

Follow it with:

```bash
tail -f ~/hrcracknet/logs/ablation.log
```

The runs go to `runs/ablation/<variant>`: `full`, `no-strip`, `no-distance`,
`no-hairline`, `no-line-prior`, `no-deep-supervision`, `convnext` and
`baseline`. At about 4.7 hours per run the study takes about a day and a half;
`EPOCHS=40` in front of the command halves that (every run then has 40 epochs,
so they stay comparable with each other but not with the 80-epoch runs). When
the last run has ended, the script draws the comparison into
`runs/ablation/comparison/`. A finished run is skipped and an interrupted one
resumed, so the same command continues the study after a failure.
`VARIANTS="full no-strip baseline"` limits it to those variants; every other
variable of `scripts/submit.sh` (`EPOCHS`, `AUG_PRESET`, `TRAIN_ARGS`, ...)
applies to every run.

A full run you trained before with `scripts/submit.sh` can serve as the
`full` variant instead of training it again. Link it into the study before
starting:

```bash
mkdir -p ~/hrcracknet/runs/ablation && ln -s "$(ls -d ~/hrcracknet/runs/2*/ | sort | tail -1)" ~/hrcracknet/runs/ablation/full
```

That links the newest run in `runs/`; check with `ls -l ~/hrcracknet/runs/ablation`
that it is the one you mean and that it used the default options.

The optimiser and loss study of the UNet++ project, on the full network
(ten runs into `runs/ablation_losses/<optimizer>_<loss>`):

```bash
cd ~/hrcracknet && STUDY=losses nohup bash scripts/submit_ablation.sh "$HOME/dataset_20261011/split_80_10_10" > logs/ablation_losses.log 2>&1 &
```

`OPTIMIZERS="adam"` or `LOSSES="dice tversky"` limit it; terms joined with `+`
make a combined loss (`LOSSES="dice+cldice"`). Its `adamw_hairline` run is the
`full` run of the first study and can be linked the same way.

The comparison can be drawn again at any time, also while runs are still
missing (they are left out):

```bash
cd ~/hrcracknet && python scripts/compare_runs.py runs/ablation --output runs/ablation/comparison
```

Download the figures and tables (PowerShell on your PC):

```powershell
scp -P 44065 -r se-st126157@asl.ait.ac.th:hrcracknet/runs/ablation/comparison "E:\STRUCTURAL ENGINEERING\thesis\crackdetection\training\runshrcracknet"
```

## Score an earlier run again

A finished run is scored again, without training, by an evaluation-only job on
its run directory:

```bash
EVALUATE_ONLY=1 bash scripts/submit.sh "$HOME/dataset_20261011/split_80_10_10" "$HOME/hrcracknet/runs/RUN"
```

It writes `test_metrics_evaluate_only.json` and the log prints, for every
source, the validation and test score at the shared threshold and at its own.
`best.pt`, `test_metrics.json` and the history stay as they were; the figures
in `plots/` are drawn again.

## Resume and update

Resume into the same run directory using its checkpoint:

```bash
RESUME=1 bash scripts/submit.sh /absolute/path/to/split_80_10_10 /absolute/path/to/existing/run
```

Pass the same variables as in the first submission (`ENCODER` and `TRAIN_ARGS`
included): a resume with another network stops before it changes the run
directory. Never submit two jobs writing into the same output directory. Cancel
a job using `scancel JOBID`. Update the repo only after active jobs finish:

```bash
git pull --ff-only
```

Retrieve results with `scp -r` from the server, as for the other projects.

ASL instructions: https://asl.ait.ac.th/package-slurm.html
Slurm examples: https://asl.ait.ac.th/package-script.html
PyTorch wheels: https://pytorch.org/get-started/previous-versions/
