# HRCrackNet architecture

Organization follows the Swin U-Net, FiLM U-Net, multi-decoder U-Net and
UNet++ projects (configs, src, scripts, slurm, docs). The data pipeline,
augmentation library, loss terms, metrics, figures and training loop are those
of the other hairline trainers; the network is new, and so is one loss term
for its distance-field output. It is not conditioned on the ground sampling
distance.

| Module | Responsibility |
| --- | --- |
| `src/hrcracknet/common.py` | Shared imports, constants (default encoder), default paths, random seeds |
| `src/hrcracknet/gsd.py` | GSD table (`pattern,gsd_mm_per_px`) and per-GSD groups, for the data pipeline and the reports |
| `src/hrcracknet/augmentations.py` | Reference augmentation, extended library, label safety, presets |
| `src/hrcracknet/data.py` | Image/mask pairing, dataset with GSD bookkeeping, augmentation preview, batching |
| `src/hrcracknet/models.py` | `HRCrackNet` (encoder, aggregation, strip context, decoder, stems, heads), EMA, checkpoint reconstruction |
| `src/hrcracknet/losses.py` | The loss terms (BCE, focal, Dice, Tversky, clDice), the distance-field term, deep supervision |
| `src/hrcracknet/evaluation.py` | Threshold sweeps, tiled inference, tolerant metrics, predictions |
| `src/hrcracknet/plots.py` | Epoch curves and final evaluation figures/tables |
| `src/hrcracknet/comparison.py` | Figures and tables comparing finished runs (the ablation and loss studies) |
| `src/hrcracknet/training.py` | Argument parsing, optimizer/scheduler, training/validation, resume |
| `src/hrcracknet/self_test.py` | The trainer's built-in checks (`--self-test`) |
| `scripts/train.py` | TOML defaults and training CLI |
| `scripts/encoder_weights.py` | Cache the pretrained encoder, or check that it is cached |
| `scripts/submit_ablation.sh` | Train every variant of a study, one job after another |
| `scripts/compare_runs.py` | Draw finished runs in one set of comparison figures |
| `slurm/train_hrcracknet.sbatch` | ASL GPU allocation and execution |

`augmentations.py`, `data.py`, `gsd.py`, `evaluation.py` and `plots.py` are
unchanged copies of the UNet++ project's files.

## What the network is built against

The UNet++ run on the 12,736-tile dataset (ConvNeXt-Tiny, test set, no TTA)
scored well on the close-range and benchmark photos and poorly on the
orthomosaic tiles of the facades:

| Test source | Tolerant F1 at the shared threshold 0.875 | At the source's own threshold |
| --- | --- | --- |
| Benchmark and close-range sets | 0.86 to 0.98 | about the same |
| EH104 (`eh#_r#_c#`) | 0.685 | 0.696 at 0.70 |
| RAMA8 west (`rama#_west-...`) | 0.443 | 0.456 at 0.02 |
| RAMA8 faces 3 to 8 (`f#_r#_c#`) | 0.293 (recall 0.19) | 0.385 at 0.02 |
| RAMA8 east (`rama#_east-...`) | 0.274 | 0.379 at 0.02 |

On the facades the network found few of the cracks and was unsure of the ones
it found (best threshold 0.02). Scoring the multi-decoder U-Net's detection of
RAMA8 face 3 against the hand-drawn crack map showed why:

1. The missed cracks are barely in the pixels: the mean dip under the drawn
   lines is about 4 grey levels, against about 13 of surface texture.
2. Most false detections are dead-straight, mostly vertical dashes on formwork
   board marks and drip stains.
3. The labels are drawn by hand and sit a pixel or two off the crack.

Each part of HRCrackNet answers one of these, or the loss of thin detail in an
encoder that halves the resolution five times:

| Problem | Part |
| --- | --- |
| A crack 1 to 2 px wide vanishes in a 1/32-scale encoder and must be recovered by the decoder | HRNet keeps a 1/4-scale branch through the whole network; the hairline stem adds full resolution |
| A faint crack (4 grey levels) is below the texture (13) at any single pixel | Strip convolutions sum along the crack's direction, where the dips line up and the texture does not |
| Formwork joints and drip stains are straight lines too | The strip-pooling gate sees whether a response runs on along the whole row or column segment |
| Hand-drawn labels a pixel off make a correct prediction a miss and a false positive | The distance field rewards a prediction near the label, graded by distance |

An architecture cannot recover a crack that is not in the image, and the
facade tiles of the training set include few crack-free facade tiles as
negatives. The ablation study shows how much of the gap each part closes.

## The network

```text
image ──┬─ HRNet-W32 (ImageNet) ── branches at 1/4 (32 ch), 1/8 (64), 1/16 (128), 1/32 (256); stem at 1/2 (64)
        └─ hairline stem: RGB + black-hat / top-hat ── 1/1 (32 ch)

 1/4   all branches upsampled to 1/4, concatenated (480 ch), 1x1 conv ─▶ 128 ch
       strip context block ─▶ strip context block ──────────────────── 1x1 ─▶ deep supervision
 1/2   upsample, + HRNet stem feature ─▶ two 3x3 conv (64 ch) ─────────── 1x1 ─▶ deep supervision
 1/1   upsample, + hairline stem ─▶ two 3x3 conv (32 ch) ─┬─ 3x3 conv ─▶ crack logits
                                                          └─ 3x3 conv ─▶ distance field (training only)
```

- **Encoder** (`--encoder`): HRNet-W32 (Wang et al., 2020) through
  [segmentation_models_pytorch](https://github.com/qubvel-org/segmentation_models.pytorch)
  (`tu-hrnet_w32`, a timm model), ImageNet-pretrained, 29.3 M parameters. A
  U-Net encoder halves the resolution stage after stage and the decoder has to
  recover thin cracks from coarse features; HRNet runs four branches at 1/4 to
  1/32 scale side by side and exchanges information between them at every
  stage, so the 1/4-scale branch is as deep as the others. HrSegNet (Li et al.,
  2023) uses this principle for cracks. The network takes the outputs of the
  four branches (timm's `feature_location=""`), as HRNetV2 does for
  segmentation, not timm's default classification features, which pass each
  branch through a bottleneck of 128 to 1024 channels (1.6 M parameters more).
  The image is normalised inside the model with the statistics of the
  encoder's pretraining, so the model is called with RGB in [0, 1].
  `tu-hrnet_w18` (10.1 M parameters in all) and `tu-hrnet_w48` are the smaller
  and the larger HRNet; any other smp encoder works with the same decoder.
- **Aggregation** (HRNetV2): the four branch outputs are brought to 1/4 scale,
  concatenated and reduced to `--context-channels` (128) by a 1x1 convolution.
- **Strip context** (`--strip-blocks`, 2): residual blocks at 1/4 scale. In
  each, a 1x1 convolution halves the channels, four depthwise strip
  convolutions of `--strip-length` (9) pixels run along the horizontal, the
  vertical and both diagonals (the strip convolution of CoANet, Mei et al.,
  2021, made for roads in aerial images), and a 1x1 convolution joins them.
  Nine pixels at 1/4 scale cover 36 pixels of the image, two blocks about 70.
  The result is weighted by a gate before it is added to the block's input:
  the strip pooling of SPNet (Hou et al., 2020), the mean over a row segment
  and a column segment of `--pool-length` (31) pixels, a 3-pixel convolution
  across each, their sum, a 1x1 convolution and a sigmoid. SPNet pools whole
  rows and columns; here the segments have a fixed length (124 image pixels),
  because the network trains on 256-px crops and is evaluated on tiles of up
  to 1024 px, and a mean over the whole row would change with the input size.
  A test checks that a pixel away from the border gets the same output in a
  crop as in the larger tile.
- **Decoder**: upsampling to 1/2 scale, concatenation with HRNet's own
  1/2-scale stem feature, two 3x3 convolutions (`--decoder-channels`, 64);
  upsampling to full resolution, concatenation with the hairline stem, two
  3x3 convolutions (32); a 3x3 output convolution. An encoder without
  1/2-scale features (ConvNeXt) gets a 1/2-scale stem of two 3x3 convolutions
  on the image, the first with stride 2 (`--half-channels`, 48), as in the
  UNet++ project.
- **Hairline stem**, as in the other trainers: two 3x3 convolutions at full
  resolution on the image and on fixed black-hat and top-hat responses
  (kernels 7 and 15), which respond to dark and to bright lines narrower than
  the kernel. `--no-line-prior` drops the thin-line channels,
  `--no-hairline-stem` the whole stem.
- **Deep supervision**: 1x1 crack outputs on the 1/4-scale context and on the
  1/2-scale decoder features, upsampled to full size and trained on the crack
  mask. `--no-deep-supervision` removes them. They are used in training only.
- **Distance-field head**: a 3x3 convolution on the full-resolution features
  regresses how close each pixel is to a crack (below). `--no-distance-head`
  removes it. Used in training only; the network's output is the crack logits
  alone, so evaluation, thresholds and inference are as for the other models.

All outputs start at the crack prior (bias for a probability of 0.03).

| Network | Parameters |
| --- | --- |
| HRCrackNet on HRNet-W32, default | 29.9 M |
| HRCrackNet on ConvNeXt-Tiny (the ablation run) | 28.5 M |
| HRCrackNet on HRNet-W18 | 10.1 M |
| UNet++ on ConvNeXt-Tiny | 35.1 M |
| Multi-decoder U-Net on ResNet-34, three decoders | 31.0 M |

On a CPU one training step (4 crops of 256 px) took 6.0 s for HRCrackNet and
7.5 s for UNet++; the time per epoch on the GPU shows in the first run's log.

## Loss

```text
loss = L(output, mask)                                  the --loss terms
     + 0.2 R(output at 1/4, mask) + 0.2 R(output at 1/2, mask)   their region terms (all but clDice)
     + 0.5 D(distance output, mask)                          the distance-field term
```

`--loss` names one or more terms, added with their `--<term>-weight`:

| Term | What it measures | Weight |
| --- | --- | --- |
| `bce` | Binary cross-entropy per pixel | 1 |
| `focal` | Binary focal loss: cross-entropy turned down on pixels already right (alpha 0.25, gamma 2) | 1 |
| `dice` | 1 - soft Dice over the batch | 1 |
| `tversky` | 1 - Tversky index, misses weighted 0.6 and false positives 0.4 | 1 |
| `cldice` | Soft clDice: continuity of the crack centreline, ramped in over 5 epochs | 0.5 |

The default, `hairline`, is BCE + Tversky + clDice, the loss of the other
projects. The deep-supervision weights are `--aux-weights` (0.2 0.2).

**The distance field.** For every pixel, the target is 1 on the crack and
falls by 1/6 per pixel of distance from it, to 0 at 6 pixels
(`--distance-margin` 5). It is built inside the loss from the mask by
repeated 3x3 max pooling, so it follows every augmentation and the shared data
pipeline stays unchanged. The term `D` is the mean absolute difference between
the sigmoid of the distance output and the field, taken separately over the
pixels within the margin and over the others and averaged with equal weight,
so the few pixels around a hairline count as much as the wall
(`--distance-weight` 0.5). A crack mask one or two pixels wide gives few
positive pixels, and a label drawn a pixel off turns all of them into misses;
the field changes gradually, so a prediction one pixel off still earns most of
its reward (deep distance transform, Wang et al., 2020). Pixels the label
safety check removed from the loss are left out of this term too; the
tolerance ring of `--label-tolerance` is not needed for it.

`training_history.csv` records the whole objective (`train_loss`), the part
of the network output (`train_output_loss`, the terms `val_loss` has) and the
distance term before its weight (`train_distance_loss`).

## Ablation study

`scripts/submit_ablation.sh` trains one run per variant, with the same data,
augmentation, loss and schedule, one job after another:

| Variant | Network | What it measures |
| --- | --- | --- |
| `full` | Everything above | The reference |
| `no-strip` | `--no-strip-context` | The strip convolutions and the pooling gate |
| `no-distance` | `--no-distance-head` | The distance-field target |
| `no-hairline` | `--no-hairline-stem` | Full-resolution detail (the thin-line prior goes with it) |
| `no-line-prior` | `--no-line-prior` | The fixed black-hat / top-hat channels alone |
| `no-deep-supervision` | `--no-deep-supervision` | The 1/4- and 1/2-scale outputs |
| `convnext` | `--encoder tu-convnext_tiny` | HRNet against the encoder of the UNet++ runs, same decoder |
| `baseline` | All four parts off | HRNet-W32, aggregation and decoder: the total gain of the parts |

`STUDY=losses` trains the optimiser and loss grid of the UNet++ project on the
full network instead (AdamW and Adam x BCE, Focal, Dice, Tversky and BCE +
Tversky + clDice).

`scripts/compare_runs.py` reads the finished runs of a study and writes, to
`runs/ablation/comparison/`:

- `comparison_test_metrics.png`: tolerant F1, F1 and IoU on the test set, one
  row per run, with the full network as a dashed line;
- `comparison_sources.png`: the tolerant F1 of every run on every test source,
  each source at its own validation-calibrated threshold, which shows which
  part helps the facade tiles and which the photos;
- `comparison_validation_curves.png`: validation tolerant F1 per epoch, one
  panel per run, the full network in grey in each;
- `comparison_precision_recall.png`: tolerant precision against recall, with
  curves of equal tolerant F1;
- `comparison_heatmap.png`: the optimiser x loss grid, for the loss study only;
- `comparison_summary.csv` (every metric, the change of tolerant F1 against the
  full network, the tolerant F1 at the per-source thresholds and per source)
  and `comparison_summary.md` (a table to paste).

A run is named by what sets it apart, read from its `configuration.json`.
Every run is evaluated at its own validation-calibrated threshold. The loss
values themselves are not compared: the terms have different scales.

## Thresholds

As in the UNet++ project: the threshold is calibrated on validation (2-px
tolerant F1, grid 0.01 to 0.99), and every source family with at least
`--source-threshold-min-images` (10) validation images also gets its own.
`test_metrics.json` reports the test metrics at the shared threshold and,
under `with_source_thresholds`, with every source at its own; `best.pt` stores
both (`optimal_threshold`, `source_thresholds` and their `_tta` variants).
`--evaluate-only` (`EVALUATE_ONLY=1` on the server) scores a `best.pt` again
and writes `test_metrics_evaluate_only.json`.

## Training settings

The settings are those of the other CNN trainers, so that the runs differ in
the network only: AdamW, learning rate 3e-4 with the pretrained encoder at a
tenth of it, weight decay 1e-4, two warm-up epochs and cosine decay, EMA
weights, mixed precision, gradient clipping at 1, crops of 256 px, batch 12,
80 epochs with early stopping after 15 epochs without improvement. The
aggregation, strip context, decoder, stems and outputs train at the full rate.
HRNet has no stochastic depth; `--drop-path-rate` (0.1) applies to the
ConvNeXt run.

## Augmentation

The augmentation library of the earlier trainers is used whole: the reference
augmentation plus 63 registered operations (mixing, synthetic cracks and
distractors, rescaling, warps, orthomosaic artefacts, photometric, weather,
blur, noise, compression, occlusion), RandAugment, the label-safety check and
the augmentation preview. The default preset is `all`: every operation at
probability 0.05 and magnitude 0.75 on top of the reference augmentation.
`--aug-preset heavy` is the strongest preset without the seven operations that
can harm crack labels, and `--aug-preset reference` is the augmentation the
Swin and FiLM U-Net runs used. The table of probabilities is in
`python scripts/train.py --help`.

## The GSD table

The network has no GSD input. `configs/gsd.csv` (the pixel size of each
source) is used by the data pipeline only: the mixing augmentations (mosaic,
CutMix, MixUp, copy-paste) combine tiles of one source and one GSD, the
augmentation preview shows the GSD of each tile, and the test metrics are also
reported per GSD value.

## Checkpoints

`last.pt` and `best.pt` are written to a temporary file and renamed. `best.pt`
stores the architecture name (`hrcracknet`) and arguments (encoder, context
channels, strip blocks and lengths, decoder channels, stems, deep supervision,
distance head), the GSD table, the augmentation configuration and the shared
and per-source thresholds, and `hrcracknet.models.load_model_from_checkpoint`
rebuilds the network from them without downloading anything. `--init-weights`
and `--resume` need the same network as the checkpoint and stop with the
differing options otherwise.

## Inference

There is no full-orthomosaic inference for this model yet. A model loaded with
`load_model_from_checkpoint` is called as `model(images)` with RGB in [0, 1]
and a height and width that are multiples of 32, and returns the crack logits,
like the UNet++ model; `checkpoint["optimal_threshold"]` is the threshold
calibrated on validation and `checkpoint["source_thresholds"]` maps every
source family to its own.

## References

- Wang, J. et al. (2020). Deep high-resolution representation learning for visual recognition. IEEE TPAMI.
- Li, Y. et al. (2023). Real-time high-resolution neural network with semantic guidance for crack segmentation (HrSegNet). Automation in Construction.
- Mei, J. et al. (2021). CoANet: Connectivity attention network for road extraction from satellite imagery. IEEE TIP.
- Hou, Q. et al. (2020). Strip pooling: Rethinking spatial pooling for scene parsing. CVPR.
- Wang, Y. et al. (2020). Deep distance transform for tubular structure segmentation in CT scans. CVPR.
- Shit, S. et al. (2021). clDice: a novel topology-preserving loss function for tubular structure segmentation. CVPR.
