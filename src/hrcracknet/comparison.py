'Comparison of finished training runs: the ablation study of the network parts, and the optimiser and loss study.'

from __future__ import annotations
from .common import DEFAULT_ENCODER, Path, csv, dataclass, json, math, np
from .losses import LOSS_ALIASES, LOSS_TERMS, loss_label, loss_terms
from .plots import BLUE_RAMP, PLOT_INK, PLOT_STYLE, SERIES_COLOURS, read_history

# --------------------------------------------------------------------------- #
# Runs compared side by side. Each run is a row, named by what sets it apart (the parts of the
# network it lacks, its encoder, its loss), and a series, its optimiser: the same colour means
# the same optimiser in every figure. The reference is the full network with the default loss
# and AdamW; the other runs are measured against it.
# --------------------------------------------------------------------------- #
OPTIMIZER_LABELS = {"adamw": "AdamW", "adam": "Adam"}
OPTIMIZER_COLOURS = {"adamw": SERIES_COLOURS[0], "adam": SERIES_COLOURS[1]}
# The parts of the network that the study takes out, as (configuration.json flag, label), in the order shown.
PARTS = (("no_strip_context", "strip context"), ("no_distance_head", "distance head"),
         ("no_hairline_stem", "hairline stem"), ("no_line_prior", "thin-line prior"),
         ("no_deep_supervision", "deep supervision"))
PART_LABELS = dict(PARTS)
# Without the hairline stem there is no thin-line prior either, so these four are every part.
BASELINE = frozenset(("no_strip_context", "no_distance_head", "no_hairline_stem", "no_deep_supervision"))
ENCODER_LABELS = {"tu-hrnet_w32": "HRNet-W32", "tu-hrnet_w18": "HRNet-W18", "tu-hrnet_w48": "HRNet-W48",
                  "tu-convnext_tiny": "ConvNeXt-T", "efficientnet-b4": "EfficientNet-B4"}
# Test metrics shown in the figures, as (key in test_metrics.json, label).
FIGURE_METRICS = (("tol_f1", "Tolerant F1 (2 px)"), ("f1", "F1"), ("iou", "IoU"))
SUMMARY_METRICS = ("precision", "recall", "f1", "iou", "tol_precision", "tol_recall", "tol_f1")


def removed_parts(configuration: dict) -> tuple:
    """The parts of the network a run was trained without, in PARTS order."""
    removed = {key for key, _ in PARTS if configuration.get(key)}
    if configuration.get("strip_blocks") == 0:
        removed.add("no_strip_context")
    if "no_hairline_stem" in removed:
        removed.discard("no_line_prior")  # implied: the prior is an input of the stem
    return tuple(key for key, _ in PARTS if key in removed)


def network_label(encoder: str, removed) -> str:
    """'Full network', 'Without strip context', 'ConvNeXt-T encoder', 'Baseline (no parts)', ..."""
    removed = tuple(removed)
    if BASELINE <= set(removed):
        text = "Baseline (no parts)"
    elif removed:
        text = "Without " + ", ".join(PART_LABELS[key] for key in removed)
    else:
        text = "Full network"
    if encoder != DEFAULT_ENCODER:
        name = ENCODER_LABELS.get(encoder, encoder)
        text = f"{name} encoder" if text == "Full network" else f"{name} encoder, {text[0].lower()}{text[1:]}"
    return text


@dataclass(eq=False)  # runs are compared by identity: their history holds arrays
class Run:
    """What the comparison reads from one run directory."""
    path: Path
    optimizer: str
    terms: tuple
    encoder: str
    removed: tuple          # removed_parts() of its configuration
    best_epoch: int
    test: dict              # the block of test_metrics.json for the chosen mode
    history: dict           # training_history.csv as {column: values}
    row: str = ""           # set by compare_runs: what the figures name the run by

    @property
    def loss(self) -> str:
        return loss_label(self.terms)

    @property
    def network(self) -> str:
        return network_label(self.encoder, self.removed)

    @property
    def series(self) -> str:
        return OPTIMIZER_LABELS.get(self.optimizer, self.optimizer)

    @property
    def label(self) -> str:
        return f"{self.series}, {self.row}"

    @property
    def is_reference(self) -> bool:
        return (self.network == "Full network" and self.terms == loss_terms("hairline")
                and self.optimizer == "adamw")


def run_terms(configuration: dict) -> tuple:
    """The loss terms of a run. Runs before the term list stored the name of their loss:
    "hairline" (BCE + Tversky + clDice) or "focal" (focal + clDice)."""
    loss = configuration.get("loss", "hairline")
    if loss == "focal":
        return ("focal", "cldice")
    return loss_terms(loss)


def run_directories(paths) -> list[Path]:
    """Run directories among `paths`: a path holding configuration.json is a run, any other
    directory is searched one level down (the folder of a whole study)."""
    found = []
    for path in map(Path, paths):
        if (path / "configuration.json").is_file():
            found.append(path)
        elif path.is_dir():
            found.extend(sorted(child for child in path.iterdir() if (child / "configuration.json").is_file()))
    return found


def load_run(path: Path, mode: str = "tta") -> Run | None:
    """The run in `path`, or None while it has no test metrics (still training, or stopped)."""
    if not (path / "test_metrics.json").is_file():
        return None
    configuration = json.loads((path / "configuration.json").read_text(encoding="utf-8"))
    metrics = json.loads((path / "test_metrics.json").read_text(encoding="utf-8"))
    test = metrics.get(mode) or metrics.get("no_tta")
    return Run(path, configuration.get("optimizer", "adamw"), run_terms(configuration),
               configuration.get("encoder", DEFAULT_ENCODER), removed_parts(configuration),
               int(metrics.get("best_epoch", 0)), test, read_history(path / "training_history.csv"))


def loss_order(terms: tuple) -> tuple:
    """Single terms first, in LOSS_TERMS order, then the combinations by their number of terms."""
    return len(terms), [LOSS_TERMS.index(term) for term in terms]


def network_order(run: Run) -> tuple:
    """The full network first, then one part removed at a time, then other encoders, the baseline last."""
    keys = [key for key, _ in PARTS]
    return (BASELINE <= set(run.removed), run.encoder != DEFAULT_ENCODER, len(run.removed),
            [keys.index(key) for key in run.removed], run.encoder)


def name_rows(runs: list[Run]) -> None:
    """Name each run by what varies in the study: the network, the loss, or both."""
    networks, losses = {run.network for run in runs}, {run.terms for run in runs}
    for run in runs:
        if len(networks) == 1 and len(losses) > 1:
            run.row = run.loss
        elif len(losses) == 1 or run.terms == LOSS_ALIASES["hairline"]:
            run.row = run.network
        else:
            run.row = f"{run.network}, {run.loss}"


def write_summary(runs: list[Run], folder: Path) -> list[str]:
    """comparison_summary.csv (every metric, the change against the reference run and the tolerant F1
    of every source) and comparison_summary.md (a table to paste)."""
    sources = sorted({source for run in runs for source in run.test.get("per_source", {})})
    reference = next((run for run in runs if run.is_reference), None)
    header = ["run", "optimizer", "network", "loss", "best_epoch", "threshold", *SUMMARY_METRICS,
              "tol_f1_change", "tol_f1_source_thresholds", *(f"tol_f1 {source}" for source in sources)]
    rows = []
    for run in runs:
        own = run.test.get("with_source_thresholds", {})
        change = "" if reference is None else f"{run.test['tol_f1'] - reference.test['tol_f1']:+.4f}"
        rows.append([run.path.name, run.series, run.network, run.loss, run.best_epoch,
                     f"{run.test['threshold']:.3f}", *(f"{run.test[name]:.4f}" for name in SUMMARY_METRICS),
                     change, f"{own['tol_f1']:.4f}" if "tol_f1" in own else "",
                     *(f"{run.test['per_source'][source]['tol_f1']:.4f}" if source in run.test.get("per_source", {})
                       else "" for source in sources)])
    with (folder / "comparison_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerows([header, *rows])
    best = max(runs, key=lambda run: run.test["tol_f1"])
    several = len({run.optimizer for run in runs}) > 1
    lines = [f"| {'Optimiser | ' if several else ''}Run | Precision | Recall | F1 | IoU | Tolerant F1 | "
             f"Change | Best epoch |",
             f"| {'--- | ' if several else ''}--- | --- | --- | --- | --- | --- | --- | --- |"]
    for run in sorted(runs, key=lambda run: -run.test["tol_f1"]):
        mark = "**" if run is best else ""
        change = ("reference" if run is reference else "" if reference is None
                  else f"{run.test['tol_f1'] - reference.test['tol_f1']:+.3f}")
        lines.append(f"| {run.series + ' | ' if several else ''}{run.row} | "
                     + " | ".join(f"{run.test[name]:.3f}" for name in ("precision", "recall", "f1", "iou"))
                     + f" | {mark}{run.test['tol_f1']:.3f}{mark} | {change} | {run.best_epoch} |")
    if reference is not None:
        lines.append("\nChange: tolerant F1 against the full network with BCE + Tversky + clDice and AdamW.")
    (folder / "comparison_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return ["comparison_summary.csv", "comparison_summary.md"]


def save_comparison_figures(runs: list[Run], folder: Path, mode: str) -> list[str]:
    """The figures of the study, in the style of plots.save_plots(); returns the file names."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator

    ink, written = PLOT_INK, []
    rows = list(dict.fromkeys(run.row for run in runs))  # runs arrive sorted, so rows are in study order
    optimizers = [name for name in OPTIMIZER_LABELS if any(run.optimizer == name for run in runs)]
    optimizers += sorted({run.optimizer for run in runs} - set(optimizers))
    colour = {name: OPTIMIZER_COLOURS.get(name, ink["secondary"]) for name in optimizers}
    by_cell = {(run.optimizer, run.row): run for run in runs}
    reference = next((run for run in runs if run.is_reference), None)
    tta_note = "flip TTA" if mode == "tta" else "no TTA"
    colour_map = LinearSegmentedColormap.from_list("study_blue", BLUE_RAMP)

    def finish(figure, name):
        figure.savefig(folder / name)
        plt.close(figure)
        written.append(name)

    def legend_handles():
        return [Line2D([], [], marker="o", linestyle="", markersize=8, color=colour[name],
                       markeredgecolor="white", markeredgewidth=1.5, label=OPTIMIZER_LABELS.get(name, name))
                for name in optimizers]

    def value_grid(axis, grid, row_labels, column_labels, title):
        """A heatmap with every value written in its cell; empty cells say 'not run'."""
        low, high = np.nanmin(grid), np.nanmax(grid)
        image = axis.imshow(grid, cmap=colour_map, vmin=low, vmax=high if high > low else low + 1e-3, aspect="auto")
        for row in range(grid.shape[0]):
            for column in range(grid.shape[1]):
                value = grid[row, column]
                if math.isnan(value):
                    axis.text(column, row, "not run", ha="center", va="center", fontsize=8, color=ink["muted"])
                    continue
                share = (value - low) / (high - low) if high > low else 1.0
                axis.text(column, row, f"{value:.3f}", ha="center", va="center", fontsize=9,
                          fontweight="semibold" if value == np.nanmax(grid[:, column]) else "normal",
                          color="white" if share > 0.55 else ink["primary"])
        axis.set_xticks(range(len(column_labels)), column_labels)
        axis.set_yticks(range(len(row_labels)), row_labels)
        axis.set_title(title)
        axis.grid(False)
        axis.set_xticks(np.arange(len(column_labels) - 1) + 0.5, minor=True)
        axis.set_yticks(np.arange(len(row_labels) - 1) + 0.5, minor=True)
        axis.grid(which="minor", color="white", linewidth=2)  # surface gap between cells
        axis.tick_params(which="both", length=0, labelcolor=ink["secondary"])
        for spine in axis.spines.values():
            spine.set_visible(False)
        bar = axis.figure.colorbar(image, ax=axis, fraction=0.03, pad=0.02)
        bar.outline.set_visible(False)
        bar.ax.tick_params(labelsize=8, colors=ink["muted"])

    with plt.rc_context(PLOT_STYLE):
        # 1. Test metrics: one row per run, one dot per optimiser, the reference as a dashed line.
        #    Dots, not bars, so the axis can start near the values: the runs differ by hundredths.
        height = max(3.2, 0.55 * len(rows) + 1.6)
        figure, axes = plt.subplots(1, len(FIGURE_METRICS), figsize=(4.3 * len(FIGURE_METRICS), height), sharey=True)
        positions = {row: index for index, row in enumerate(reversed(rows))}
        offsets = {name: (index - (len(optimizers) - 1) / 2) * 0.16 for index, name in enumerate(optimizers)}
        for axis, (key, title) in zip(axes, FIGURE_METRICS):
            values = [run.test[key] for run in runs]
            if reference is not None and len(runs) > 1:
                axis.axvline(reference.test[key], color=ink["axis"], linestyle="--", linewidth=1.2, zorder=1)
            for row in rows:  # a hairline joins the optimisers of one row
                cell = [by_cell[(name, row)].test[key] for name in optimizers if (name, row) in by_cell]
                if len(cell) > 1:
                    axis.plot([min(cell), max(cell)], [positions[row]] * 2, color=ink["grid"], linewidth=2, zorder=1)
            for run in runs:
                axis.plot(run.test[key], positions[run.row] + offsets[run.optimizer], "o", markersize=9,
                          color=colour[run.optimizer], markeredgecolor="white", markeredgewidth=1.5, zorder=3)
            best = max(runs, key=lambda run: run.test[key])
            axis.annotate(f"{best.test[key]:.3f}", (best.test[key], positions[best.row] + offsets[best.optimizer]),
                          xytext=(8, 0), textcoords="offset points", va="center", fontsize=8.5,
                          color=ink["secondary"])
            span = max(values) - min(values)
            pad = max(0.01, 0.15 * span)
            axis.set_xlim(min(values) - pad, max(values) + 2.2 * pad)
            axis.set_title(title)
            axis.grid(axis="y", visible=False)
            axis.xaxis.set_major_locator(MaxNLocator(5))
        axes[0].set_yticks(list(positions.values()), list(positions))
        axes[0].set_ylim(-0.6, len(rows) - 0.4)
        if len(optimizers) > 1:
            axes[-1].legend(handles=legend_handles(), loc="lower right", bbox_to_anchor=(1.0, 1.08),
                            ncol=len(optimizers), borderaxespad=0, handletextpad=0.3, columnspacing=1.2)
        figure.suptitle(f"Test metrics of every run ({tta_note}, at each run's calibrated threshold)",
                        x=0.02, ha="left", fontsize=12, fontweight="semibold")
        figure.subplots_adjust(top=1 - 0.95 / height)  # the same room for the heading at any number of rows
        note = "The best run of each metric is labelled"
        if reference is not None and len(runs) > 1:
            note += "; the dashed line is the full network (BCE + Tversky + clDice, AdamW)"
        figure.text(0.02, -0.02, note + ". Strict F1 and IoU count pixels exactly; tolerant F1 accepts a "
                                        "prediction within 2 px of a label.", fontsize=8, color=ink["muted"])
        finish(figure, "comparison_test_metrics.png")

        # 2. Every source: the tolerant F1 of each run on each source family, at the source's own
        #    validation-calibrated threshold. Shows which part helps which kind of image.
        sources = sorted({source for run in runs for source in run.test.get("per_source", {})})
        if sources:
            def source_value(run, source):
                own = run.test.get("with_source_thresholds", {}).get("per_source", {})
                block = own.get(source) or run.test.get("per_source", {}).get(source)
                return block["tol_f1"] if block else np.nan

            labels = [run.label if len(optimizers) > 1 else run.row for run in runs]
            grid = np.array([[source_value(run, source) for source in sources] for run in runs])
            figure, axis = plt.subplots(figsize=(max(7.0, 0.95 * len(sources) + 3.5), 0.48 * len(runs) + 2.2))
            value_grid(axis, grid, labels, sources, "Tolerant F1 per source (bold: the best run of each source)")
            axis.tick_params(axis="x", labelrotation=35)
            for tick in axis.get_xticklabels():
                tick.set_horizontalalignment("right")
            figure.suptitle(f"Every run on every test source ({tta_note}, each source at its own threshold)",
                            x=0.02, ha="left", fontsize=12, fontweight="semibold")
            figure.tight_layout()
            finish(figure, "comparison_sources.png")

        # 3. Optimiser x row grid, when the study has more than one optimiser (the loss study).
        if len(optimizers) > 1:
            panels = (("tol_f1", "Tolerant F1 (2 px)"), ("f1", "F1"))
            figure, axes = plt.subplots(len(panels), 1, figsize=(max(6.5, 1.55 * len(rows) + 2.2),
                                                                 1.15 * len(optimizers) * len(panels) + 1.8))
            for axis, (key, title) in zip(np.atleast_1d(axes), panels):
                grid = np.array([[by_cell[(name, row)].test[key] if (name, row) in by_cell else np.nan
                                  for row in rows] for name in optimizers])
                value_grid(axis, grid, [OPTIMIZER_LABELS.get(name, name) for name in optimizers], rows, title)
            figure.suptitle(f"Every run of the study on the test set ({tta_note})", x=0.02, ha="left", fontsize=12,
                            fontweight="semibold")
            figure.tight_layout()
            finish(figure, "comparison_heatmap.png")

        # 4. Validation curves: one panel per row, one line per optimiser, the reference in grey.
        with_history = [run for run in runs if "epoch" in run.history and "tol_f1" in run.history]
        if with_history:
            columns = min(3, len(rows))
            panel_rows = math.ceil(len(rows) / columns)
            figure, axes = plt.subplots(panel_rows, columns, figsize=(4.4 * columns, 3.0 * panel_rows + 0.9),
                                        sharey=True, squeeze=False)
            for axis, row in zip(axes.flat, rows):
                if reference is not None and reference in with_history and reference.row != row:
                    axis.plot(reference.history["epoch"], reference.history["tol_f1"], color=ink["axis"],
                              linewidth=1.5, zorder=1)
                for name in optimizers:
                    run = by_cell.get((name, row))
                    if run is None or run not in with_history:
                        continue
                    axis.plot(run.history["epoch"], run.history["tol_f1"], color=colour[name], zorder=2)
                    if run.best_epoch in run.history["epoch"]:
                        index = int(np.where(run.history["epoch"] == run.best_epoch)[0][0])
                        axis.plot(run.best_epoch, run.history["tol_f1"][index], "o", markersize=7, color=colour[name],
                                  markeredgecolor="white", markeredgewidth=1.5, zorder=4)
                axis.set_title(row, fontsize=10)
                axis.xaxis.set_major_locator(MaxNLocator(integer=True))
            for axis in axes.flat[len(rows):]:
                axis.axis("off")
            for axis in axes[:, 0]:
                axis.set_ylabel("Validation tolerant F1")
            for axis in axes[-1]:
                axis.set_xlabel("Epoch")
            handles = [Line2D([], [], color=colour[name], label=OPTIMIZER_LABELS.get(name, name)) for name in optimizers]
            if reference is not None and reference in with_history:
                handles.append(Line2D([], [], color=ink["axis"], linewidth=1.5, label="Full network"))
            axes.flat[min(columns, len(rows)) - 1].legend(handles=handles, loc="lower right", bbox_to_anchor=(1.0, 1.12),
                                                          ncol=len(handles), borderaxespad=0)
            figure.suptitle("Validation tolerant F1 per epoch (dots: the epoch kept as best.pt)", x=0.02, ha="left",
                            fontsize=12, fontweight="semibold")
            figure.tight_layout()
            finish(figure, "comparison_validation_curves.png")

        # 5. Precision against recall: what each run trades. Faint curves join equal tolerant F1.
        figure, axis = plt.subplots(figsize=(7.6, 5.6))
        precision = np.array([run.test["tol_precision"] for run in runs])
        recall = np.array([run.test["tol_recall"] for run in runs])
        span = max(np.ptp(precision), np.ptp(recall), 0.02)
        pad = 0.3 * span
        x_low, x_high = recall.min() - pad, min(1.0, recall.max() + 2.5 * pad)  # room for the labels
        y_low, y_high = precision.min() - pad, min(1.0, precision.max() + pad)
        f1 = lambda p, r: 2 * p * r / (p + r)
        f_low, f_high = f1(y_low, x_low), f1(y_high, x_high)
        step = next(value for value in (0.005, 0.01, 0.02, 0.025, 0.05, 0.1) if (f_high - f_low) / value <= 6)
        grid = np.linspace(x_low, x_high, 400)
        for level in np.arange(math.ceil(f_low / step) * step, f_high, step):
            with np.errstate(divide="ignore", invalid="ignore"):
                curve = level * grid / (2 * grid - level)
            visible = (2 * grid > level) & (curve >= y_low) & (curve <= y_high)
            if visible.sum() < 2:
                continue
            axis.plot(grid[visible], curve[visible], color=ink["grid"], linewidth=1.0, zorder=1)
            last = np.where(visible)[0][-1]
            axis.annotate(f"{level:.3g}", (grid[last], curve[last]), xytext=(2, 2), textcoords="offset points",
                          fontsize=7.5, color=ink["muted"])
        for run in runs:
            axis.plot(run.test["tol_recall"], run.test["tol_precision"], "o", markersize=9, color=colour[run.optimizer],
                      markeredgecolor="white", markeredgewidth=1.5, zorder=3)
        # One label per row, right of its runs; labels that would touch move down, with a leader line.
        groups = sorted(rows, key=lambda row: -np.mean([run.test["tol_precision"] for run in runs if run.row == row]))
        gap, previous = 0.05 * (y_high - y_low), None
        for row in groups:
            members = [run for run in runs if run.row == row]
            x = max(run.test["tol_recall"] for run in members)
            y = float(np.mean([run.test["tol_precision"] for run in members]))
            label_y = y if previous is None else min(y, previous - gap)
            previous = label_y
            moved = abs(label_y - y) > gap / 2
            axis.annotate(row, (x, y), xytext=(x + 0.03 * (x_high - x_low), label_y),
                          textcoords="data", fontsize=8.5, color=ink["secondary"], va="center",
                          arrowprops=dict(arrowstyle="-", color=ink["muted"], linewidth=0.8, shrinkA=0, shrinkB=6)
                          if moved else None)
        axis.set(xlabel="Tolerant recall", ylabel="Tolerant precision", xlim=(x_low, x_high), ylim=(y_low, y_high),
                 title="Tolerant precision and recall")
        if len(optimizers) > 1:
            axis.legend(handles=legend_handles(), loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=len(optimizers),
                        borderaxespad=0.2)
        axis.text(0, -0.12, f"Test set, {tta_note}. Each dot is one run; grey curves join points of equal "
                            "tolerant F1.", transform=axis.transAxes, fontsize=8, color=ink["muted"], va="top")
        finish(figure, "comparison_precision_recall.png")
    return written


def compare_runs(paths, output: Path, mode: str = "tta") -> list[str]:
    """Read the runs in `paths` (run directories, or folders of them), write the summary tables and
    figures to `output` and return the file names. Unfinished runs are reported and left out."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    runs, seen = [], {}
    for path in run_directories(paths):
        run = load_run(path, mode)
        if run is None:
            print(f"Left out, no test metrics yet: {path}")
            continue
        key = (run.optimizer, run.terms, run.encoder, run.removed)
        if key in seen:
            print(f"Left out, the same network, loss and optimiser as {seen[key].name}: {path}")
            continue
        seen[key] = path
        runs.append(run)
    if not runs:
        raise SystemExit("No finished run (with test_metrics.json) found in " + ", ".join(map(str, paths)))
    runs.sort(key=lambda run: (network_order(run), loss_order(run.terms), list(OPTIMIZER_LABELS).index(run.optimizer)
                               if run.optimizer in OPTIMIZER_LABELS else 99))
    name_rows(runs)
    written = write_summary(runs, output)
    try:
        written += save_comparison_figures(runs, output, mode)
    except ImportError:
        print("matplotlib is not installed, so only the tables were written (python -m pip install matplotlib).")
    width = max(len(run.label) for run in runs) + 2
    for run in runs:
        print(f"{run.label:{width}s} tolerant F1 {run.test['tol_f1']:.4f}  F1 {run.test['f1']:.4f}  "
              f"IoU {run.test['iou']:.4f}  (best epoch {run.best_epoch}, {run.path.name})")
    return written


__all__ = [
    'OPTIMIZER_LABELS', 'OPTIMIZER_COLOURS', 'PARTS', 'PART_LABELS', 'BASELINE', 'ENCODER_LABELS', 'FIGURE_METRICS',
    'SUMMARY_METRICS', 'removed_parts', 'network_label', 'Run', 'run_terms', 'run_directories', 'load_run',
    'loss_order', 'network_order', 'name_rows', 'write_summary', 'save_comparison_figures', 'compare_runs',
]
