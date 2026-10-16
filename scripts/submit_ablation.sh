#!/bin/bash
# Ablation study: one training run per variant, one job after another, then the comparison figures.
# ASL has removed a second job of the same user from the queue while the first was running, so a
# job is submitted only when the queue is empty. Start it in the background on the login node; it
# survives the SSH session:
#
#   nohup bash scripts/submit_ablation.sh "$HOME/dataset_20261011/split_80_10_10" > logs/ablation.log 2>&1 &
#
# STUDY=components (default) takes the network apart, into runs/ablation/<variant>:
#   full                 the whole network (HRNet-W32, strip context, distance head, hairline stem
#                        with thin-line prior, deep supervision)
#   no-strip             without the strip context
#   no-distance          without the distance-field head
#   no-hairline          without the hairline stem (and so without the thin-line prior)
#   no-line-prior        the hairline stem on RGB only
#   no-deep-supervision  without the 1/4- and 1/2-scale outputs
#   convnext             the whole network on ConvNeXt-Tiny, the encoder of the UNet++ runs
#   baseline             HRNet-W32, aggregation and decoder only: none of the parts
# STUDY=losses trains the optimiser and loss grid of the UNet++ project on the whole network,
# into runs/ablation_losses/<optimizer>_<loss>.
#
# A run that has finished (test_metrics.json) is skipped and a run with a last.pt is resumed, so
# the same command continues the study after an interruption. Every other variable of
# scripts/submit.sh (EPOCHS, AUG_PRESET, TRAIN_ARGS, ...) applies to every run.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ $# -ne 1 ]]; then
    echo "Usage: bash scripts/submit_ablation.sh /absolute/dataset/path" >&2
    exit 2
fi
DATASET="$1"
STUDY="${STUDY:-components}"
VARIANTS="${VARIANTS:-full no-strip no-distance no-hairline no-line-prior no-deep-supervision convnext baseline}"
OPTIMIZERS="${OPTIMIZERS:-adamw adam}"
# One loss per word: a term alone, "hairline" (BCE + Tversky + clDice), or terms joined by "+".
LOSSES="${LOSSES:-hairline bce focal dice tversky}"
ENV_DIR="${ENV_DIR:-$HOME/.conda/envs/${ENV_NAME:-swinunet}}"
case "$STUDY" in
    components) STUDY_DIR="$(realpath -m "${STUDY_DIR:-$ROOT/runs/ablation}")" ;;
    losses) STUDY_DIR="$(realpath -m "${STUDY_DIR:-$ROOT/runs/ablation_losses}")" ;;
    *) echo "STUDY takes components or losses, not: $STUDY" >&2; exit 2 ;;
esac
# The trainer options of each variant, and its encoder (empty: the default, HRNet-W32).
variant_options() {
    case "$1" in
        full) echo "" ;;
        no-strip) echo "--no-strip-context" ;;
        no-distance) echo "--no-distance-head" ;;
        no-hairline) echo "--no-hairline-stem" ;;
        no-line-prior) echo "--no-line-prior" ;;
        no-deep-supervision) echo "--no-deep-supervision" ;;
        convnext) echo "" ;;
        baseline) echo "--no-strip-context --no-distance-head --no-hairline-stem --no-deep-supervision" ;;
        *) return 1 ;;
    esac
}
variant_encoder() {
    [[ "$1" == convnext ]] && echo "tu-convnext_tiny" || echo "${ENCODER:-tu-hrnet_w32}"
}
if [[ "$STUDY" == components ]]; then
    for variant in $VARIANTS; do  # a misspelt variant would only show after the earlier runs
        variant_options "$variant" > /dev/null || {
            echo "Unknown variant: $variant (full no-strip no-distance no-hairline no-line-prior" \
                 "no-deep-supervision convnext baseline)" >&2; exit 2; }
    done
fi
mkdir -p "$STUDY_DIR" "$ROOT/logs"

stamp() { date '+%Y-%m-%d %H:%M:%S'; }
queue_empty() {
    local jobs
    jobs=$(squeue -h -u "$USER") && [[ -z "$jobs" ]]  # a failed squeue does not count as empty
}
wait_for_queue() {
    until queue_empty; do sleep 120; done
}
# submit RUN_DIR DESCRIPTION, with the run's variables in the environment
submit() {
    local run="$1" description="$2" resume=0
    if [[ -f "$run/test_metrics.json" ]]; then
        echo "$(stamp) finished already: $run"
        return
    fi
    wait_for_queue
    if [[ -f "$run/last.pt" ]]; then
        resume=1
    elif [[ -e "$run" ]]; then
        echo "$(stamp) skipped: $run exists without last.pt or test_metrics.json (did the job fail?);" \
             "remove the folder to train it again"
        return
    fi
    echo "$(stamp) submitting $description resume=$resume -> $run"
    RESUME="$resume" bash "$ROOT/scripts/submit.sh" "$DATASET" "$run" || echo "$(stamp) submission failed: $run"
    sleep 120  # let the job appear in the queue before it is checked again
}

if [[ "$STUDY" == components ]]; then
    echo "$(stamp) study: network parts [$VARIANTS] in $STUDY_DIR"
    for variant in $VARIANTS; do
        options="$(variant_options "$variant")"
        train_args="${TRAIN_ARGS:-}"
        [[ -z "$options" ]] || train_args="${train_args:+$train_args }$options"
        ENCODER="$(variant_encoder "$variant")" TRAIN_ARGS="$train_args" \
            submit "$STUDY_DIR/$variant" "variant=$variant encoder=$(variant_encoder "$variant") options='$options'"
    done
else
    echo "$(stamp) study: optimisers [$OPTIMIZERS] x losses [$LOSSES] in $STUDY_DIR"
    for optimizer in $OPTIMIZERS; do
        for loss in $LOSSES; do
            OPTIMIZER="$optimizer" LOSS="${loss//+/ }" \
                submit "$STUDY_DIR/${optimizer}_${loss//+/-}" "optimizer=$optimizer loss='${loss//+/ }'"
        done
    done
fi
wait_for_queue
echo "$(stamp) every run has ended; drawing the comparison"
LD_LIBRARY_PATH="$ENV_DIR/lib:${LD_LIBRARY_PATH:-}" MPLBACKEND=Agg \
    "$ENV_DIR/bin/python" "$ROOT/scripts/compare_runs.py" "$STUDY_DIR" --output "$STUDY_DIR/comparison"
echo "$(stamp) done"
