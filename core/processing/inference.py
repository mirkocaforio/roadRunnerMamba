"""
Inference, evaluation and reporting for the RSCDMamba model.

This module provides three things:

1. `TrainingLogParser` / `TrainingHistory`
   Rebuild the per-epoch training history from the plain-text logs written by
   `ConsoleLogger` (core/processing/observer.py) and plot train/validation loss,
   accuracy, F1, learning-rate and confidence curves.  Works on a single log
   file, a list of files, a glob, or a whole `logs/` directory (resumed runs
   produce several files that are merged by epoch number).

2. `Inference`
   Model loading, complexity/latency/memory benchmarking, and full dataset
   evaluation with extended metrics (macro/micro/weighted averages, Cohen's
   kappa, MCC, top-k accuracy, per-class precision/recall/F1/specificity,
   calibration ECE/MCE).

3. A rich plotting + export layer
   ROC, Precision-Recall, confusion matrix (raw + normalized), confidence
   histograms, per-class metric bars, class distribution, reliability diagram
   and a grid of the most confidently misclassified images.  Everything is
   saved as **PDF** by default (vector, publication ready); PNG is available
   via `--plot-format`.

All figures are written as one file per plot, with the split name as suffix
(e.g. `confusion_matrix_val.pdf`, `confusion_matrix_test.pdf`).
"""

from typing import Optional, Union, Dict, List, Tuple, Any, Sequence
import argparse
import csv
import datetime as _dt
import glob as globmod
import json
import math
import os
import re
import sys
import time

import numpy as np
import psutil
import torch
from torch.utils.data import DataLoader, ConcatDataset

# Select a non-interactive backend before pyplot is imported: training and
# evaluation normally run on a headless server.
import matplotlib
if (os.environ.get('MPLBACKEND') is None
        and sys.platform.startswith('linux')
        and not os.environ.get('DISPLAY')):
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    from torch.utils.flop_counter import FlopCounterMode
    FLOP_COUNTER_AVAILABLE = True
except ImportError:
    FLOP_COUNTER_AVAILABLE = False

from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
    cohen_kappa_score, matthews_corrcoef, top_k_accuracy_score,
    roc_curve, auc,
    precision_recall_curve, average_precision_score
)
from sklearn.preprocessing import label_binarize

# Import del modello e utilities
from core.model.net import RSCDMamba
from core.processing.checkpoint import CheckpointLoader
from core.data.dataloader import build_dataset

torch.serialization.add_safe_globals([RSCDMamba])


# ======================================================================
# Figure saving helper
# ======================================================================

PLOT_FORMATS = ('pdf', 'png', 'both')


def save_figure(
    fig,
    save_dir: Optional[str],
    basename: str,
    plot_format: str = 'pdf',
    dpi: int = 150,
    show: bool = False,
    verbose: bool = True,
) -> List[str]:
    """
    Save a matplotlib figure as PDF and/or PNG.

    Args:
        fig:         The figure to save.
        save_dir:    Destination directory (created if missing).  None -> not saved.
        basename:    File name without extension.
        plot_format: 'pdf' (default), 'png' or 'both'.
        dpi:         Raster resolution — only relevant for PNG.
        show:        Call plt.show() before closing the figure.
        verbose:     Print the written paths.

    Returns:
        List of written file paths.
    """
    if plot_format not in PLOT_FORMATS:
        raise ValueError(f"plot_format must be one of {PLOT_FORMATS}, got '{plot_format}'")

    paths: List[str] = []
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        formats = ['pdf', 'png'] if plot_format == 'both' else [plot_format]
        for fmt in formats:
            path = os.path.join(save_dir, f"{basename}.{fmt}")
            fig.savefig(path, format=fmt, dpi=dpi, bbox_inches='tight')
            paths.append(path)
            if verbose:
                print(f"  ✓ saved → {path}")

    if show:
        plt.show()
    plt.close(fig)
    return paths


# ======================================================================
# ROC / Precision-Recall curve helpers
# ======================================================================

# Line styles of the averaged curves, shared by every curve figure
MICRO_STYLE = dict(color='deeppink', lw=2.5, linestyle=':')
MACRO_STYLE = dict(color='black', lw=2.5, linestyle='--')


def compute_curve_data(
    targets: np.ndarray,
    probabilities: np.ndarray,
    num_classes: int,
) -> Optional[Dict[str, Any]]:
    """
    One-vs-rest ROC / PR curves of every class plus the micro and macro averages.

    A class with no positive (or no negative) sample in the split has no
    defined curve: it is listed in 'skipped' and left out of the averages,
    which would otherwise turn NaN.

    Returns None when no class has a defined curve, otherwise a dict with:
        'roc'      {class index: (fpr, tpr, auc)}
        'pr'       {class index: (recall, precision, average precision)}
        'micro'    {'roc': (fpr, tpr, auc), 'pr': (recall, precision, ap)}
                   every (sample, class) pair pooled into one binary problem
        'macro'    same layout, plus the 'roc_weighted' / 'pr_weighted' scores
                   (per-class scores weighted by support)
        'valid', 'skipped'  class indices
        'support'  number of samples of each class
    'micro' and 'macro' are None with two classes: the two columns are
    complementary, so the positive-class curve already says everything.
    """
    targets = np.asarray(targets)
    probabilities = np.asarray(probabilities, dtype=float)
    # label_binarize would collapse to a single column with two classes
    y_true = np.eye(num_classes, dtype=int)[targets]
    support = y_true.sum(axis=0)
    valid = [i for i in range(num_classes) if 0 < support[i] < len(targets)]
    skipped = [i for i in range(num_classes) if i not in valid]
    if not valid:
        return None

    roc: Dict[int, Tuple] = {}
    pr: Dict[int, Tuple] = {}
    for i in valid:
        fpr, tpr, _ = roc_curve(y_true[:, i], probabilities[:, i])
        roc[i] = (fpr, tpr, float(auc(fpr, tpr)))
        precision, recall, _ = precision_recall_curve(y_true[:, i], probabilities[:, i])
        pr[i] = (recall, precision,
                 float(average_precision_score(y_true[:, i], probabilities[:, i])))

    data: Dict[str, Any] = {'roc': roc, 'pr': pr, 'micro': None, 'macro': None,
                            'valid': valid, 'skipped': skipped, 'support': support}
    if num_classes == 2:
        return data

    y_flat, p_flat = y_true.ravel(), probabilities.ravel()
    fpr, tpr, _ = roc_curve(y_flat, p_flat)
    precision, recall, _ = precision_recall_curve(y_flat, p_flat)
    data['micro'] = {
        'roc': (fpr, tpr, float(auc(fpr, tpr))),
        'pr': (recall, precision, float(average_precision_score(y_flat, p_flat))),
    }

    # Macro ROC: mean TPR on the union of every class's FPR points
    all_fpr = np.unique(np.concatenate([roc[i][0] for i in valid]))
    mean_tpr = np.mean([np.interp(all_fpr, roc[i][0], roc[i][1]) for i in valid], axis=0)
    # Macro PR: mean precision on a common recall grid; its score is the
    # mean of the per-class APs (the area under an averaged PR curve is not)
    recall_grid = np.linspace(0.0, 1.0, 1001)
    mean_precision = np.mean([np.interp(recall_grid, pr[i][0][::-1], pr[i][1][::-1])
                              for i in valid], axis=0)
    weights = support[valid] / support[valid].sum()
    data['macro'] = {
        'roc': (all_fpr, mean_tpr, float(auc(all_fpr, mean_tpr))),
        'pr': (recall_grid, mean_precision, float(np.mean([pr[i][2] for i in valid]))),
        'roc_weighted': float(np.sum(weights * np.array([roc[i][2] for i in valid]))),
        'pr_weighted': float(np.sum(weights * np.array([pr[i][2] for i in valid]))),
    }
    return data


def auc_scores_from_curves(curves: Dict[str, Any], class_names: List[str]) -> Dict[str, float]:
    """
    Flatten compute_curve_data() into the 'auc_scores' dict:
    roc_auc_<class>, pr_auc_<class> and roc_auc_/pr_auc_ macro, micro, weighted.
    With two classes the averages are the positive-class (index 1) scores.
    """
    scores: Dict[str, float] = {}
    for i, name in enumerate(class_names):
        scores[f'roc_auc_{name}'] = curves['roc'][i][2] if i in curves['roc'] else float('nan')
        scores[f'pr_auc_{name}'] = curves['pr'][i][2] if i in curves['pr'] else float('nan')

    if curves['macro'] is None:
        positive = 1 if 1 in curves['roc'] else curves['valid'][0]
        for average in ('macro', 'micro', 'weighted'):
            scores[f'roc_auc_{average}'] = curves['roc'][positive][2]
            scores[f'pr_auc_{average}'] = curves['pr'][positive][2]
        return scores

    for kind in ('roc', 'pr'):
        scores[f'{kind}_auc_macro'] = curves['macro'][kind][2]
        scores[f'{kind}_auc_micro'] = curves['micro'][kind][2]
        scores[f'{kind}_auc_weighted'] = curves['macro'][f'{kind}_weighted']
    return scores


def _value_at(x: float, xs: np.ndarray, ys: np.ndarray) -> float:
    """np.interp that also accepts a decreasing `xs` (precision_recall_curve)."""
    if xs[0] > xs[-1]:
        xs, ys = xs[::-1], ys[::-1]
    return float(np.interp(x, xs, ys))


def _add_zoom_inset(ax, lines, xlim, ylim, bounds) -> None:
    """
    Magnify a corner of `ax` in an inset.  `lines` is a list of
    (x, y, plot kwargs) already drawn on `ax`; `ylim[0]` is rounded down to
    a multiple of 0.05.
    """
    inset = ax.inset_axes(bounds)
    for x, y, style in lines:
        inset.plot(x, y, **style)
    inset.set_xlim(*xlim)
    inset.set_ylim(max(0.0, np.floor(min(ylim[0], 0.9) * 20) / 20), ylim[1])
    inset.tick_params(labelsize=8)
    inset.grid(True, alpha=0.3)
    ax.indicate_inset_zoom(inset, edgecolor='0.4')


def _draw_iso_f1(ax, levels=(0.2, 0.4, 0.6, 0.8)) -> None:
    """Grey iso-F1 curves on a precision-recall axes."""
    for f1 in levels:
        recall = np.linspace(f1 / (2 - f1) + 1e-3, 1, 200)
        precision = f1 * recall / (2 * recall - f1)
        ax.plot(recall, precision, color='gray', alpha=0.3, lw=1)
        ax.annotate(f"F1={f1:.1f}", xy=(0.88, precision[-1] + 0.02),
                    fontsize=8, color='gray')
    ax.plot([], [], color='gray', alpha=0.3, lw=1, label="iso-F1 curves")


# ======================================================================
# Training log parsing
# ======================================================================

# ConsoleLogger writes raw ANSI colour codes into the log file.
_ANSI_RE   = re.compile(r'\x1b\[[0-9;]*m')
_TS_RE     = re.compile(r'^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\t')
_EPOCH_RE  = re.compile(r'Epoch\s+(\d+)\s+completed in\s+([\d.]+)s')
_METRIC_RE = re.compile(r'^\s+-\s+([\w_]+):\s*(.+)$')

_HEADER_PATTERNS = {
    'experiment':      re.compile(r'^Starting experiment:\s*(.+)$'),
    'experiment_dir':  re.compile(r'^Experiment directory:\s*(.+)$'),
    'device':          re.compile(r'^Device:\s*(.+)$'),
    'scheduler':       re.compile(r'^Using scheduler:\s*(.+)$'),
    'total_epochs':    re.compile(r'^Training epochs:\s*(.+)$'),
    'learning_rate':   re.compile(r'^Learning rate:\s*(.+)$'),
    'batch_size':      re.compile(r'^Batch size:\s*(.+)$'),
}


class TrainingHistory:
    """
    Per-epoch training history reconstructed from one or more training logs.

    Attributes:
        epochs:  1-D int array of epoch numbers (1-based, sorted, may have gaps).
        metrics: Dict metric-name -> float array aligned with `epochs`.
                 Missing values are NaN, so matplotlib naturally breaks the line.
        meta:    Header information (experiment name, device, scheduler, lr, ...).
        files:   Log files that contributed at least one epoch.
        summary: Derived facts (best epochs, gaps, total wall time).
    """

    def __init__(
        self,
        epochs: np.ndarray,
        metrics: Dict[str, np.ndarray],
        meta: Dict[str, str],
        files: List[str],
        skipped_files: List[str],
    ):
        self.epochs = epochs
        self.metrics = metrics
        self.meta = meta
        self.files = files
        self.skipped_files = skipped_files
        self.summary = self._build_summary()

    # -- access -------------------------------------------------------

    def get(self, key: str) -> Optional[np.ndarray]:
        """Return the array for `key`, or None if absent / entirely NaN."""
        arr = self.metrics.get(key)
        if arr is None or np.all(np.isnan(arr)):
            return None
        return arr

    def has(self, key: str) -> bool:
        return self.get(key) is not None

    @property
    def available_metrics(self) -> List[str]:
        return sorted(k for k in self.metrics if self.has(k))

    def __len__(self) -> int:
        return len(self.epochs)

    # -- summary ------------------------------------------------------

    def _best(self, key: str, mode: str) -> Optional[Tuple[int, float]]:
        arr = self.get(key)
        if arr is None:
            return None
        idx = int(np.nanargmin(arr) if mode == 'min' else np.nanargmax(arr))
        return int(self.epochs[idx]), float(arr[idx])

    def _build_summary(self) -> Dict[str, Any]:
        expected = set(range(int(self.epochs[0]), int(self.epochs[-1]) + 1)) if len(self.epochs) else set()
        gaps = sorted(expected - set(int(e) for e in self.epochs))

        epoch_time = self.get('epoch_time_s')
        total_hours = float(np.nansum(epoch_time)) / 3600.0 if epoch_time is not None else None

        return {
            'n_epochs': len(self.epochs),
            'first_epoch': int(self.epochs[0]) if len(self.epochs) else None,
            'last_epoch': int(self.epochs[-1]) if len(self.epochs) else None,
            'missing_epochs': gaps,
            'total_train_hours': total_hours,
            'best_val_loss': self._best('val_loss', 'min'),
            'best_train_loss': self._best('train_loss', 'min'),
            'best_val_accuracy': self._best('val_accuracy', 'max'),
            'best_val_balanced_accuracy': self._best('val_balanced_accuracy', 'max'),
            'best_val_f1_macro': self._best('val_f1_macro', 'max'),
        }

    def print_summary(self):
        s = self.summary
        print(f"\n{'='*66}")
        print("TRAINING LOG SUMMARY")
        print(f"{'='*66}")
        for path in self.files:
            print(f"  log: {path}")
        for path in self.skipped_files:
            print(f"  skipped (no epochs): {path}")
        if self.meta:
            print(f"  {'-'*62}")
            for key in ('experiment', 'device', 'scheduler', 'total_epochs',
                        'learning_rate', 'batch_size', 'model_config'):
                if key in self.meta:
                    value = self.meta[key]
                    if len(value) > 100:
                        value = value[:97] + '...'
                    print(f"  {key:<16} {value}")
        print(f"  {'-'*62}")
        print(f"  Epochs parsed:   {s['n_epochs']}  (from {s['first_epoch']} to {s['last_epoch']})")
        if s['missing_epochs']:
            shown = s['missing_epochs'][:15]
            more = '' if len(s['missing_epochs']) == len(shown) else f" (+{len(s['missing_epochs'])-len(shown)} more)"
            print(f"  Missing epochs:  {shown}{more}")
        if s['total_train_hours'] is not None:
            print(f"  Total wall time: {s['total_train_hours']:.2f} h")
        for label, key in (('Best train loss', 'best_train_loss'),
                           ('Best val loss', 'best_val_loss'),
                           ('Best val accuracy', 'best_val_accuracy'),
                           ('Best val bal. acc.', 'best_val_balanced_accuracy'),
                           ('Best val F1 macro', 'best_val_f1_macro')):
            if s[key] is not None:
                epoch, value = s[key]
                print(f"  {label:<20} {value:.6f}  @ epoch {epoch}")
        print(f"  Metrics found:   {len(self.available_metrics)}")
        print(f"{'='*66}")

    # -- export -------------------------------------------------------

    def to_csv(self, path: str) -> str:
        """Dump the whole per-epoch history to a CSV file."""
        keys = self.available_metrics
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, 'w', newline='', encoding='utf-8') as fh:
            writer = csv.writer(fh)
            writer.writerow(['epoch'] + keys)
            for i, epoch in enumerate(self.epochs):
                row = [int(epoch)]
                for key in keys:
                    value = self.metrics[key][i]
                    row.append('' if np.isnan(value) else f"{value:.12g}")
                writer.writerow(row)
        print(f"  ✓ saved → {path}")
        return path


class TrainingLogParser:
    """
    Parse the plain-text training logs produced by `ConsoleLogger`.

    Log shape (ANSI colour codes stripped)::

        2026-06-16 10:50:13\tEpoch 26 completed in 4410.9s
        \t- learning_rate: 0.000098586242
        \t- train_loss: 0.645385408150
        \t- val_accuracy: 0.941084646760
        ...
        \t- val_confusion_matrix: [[6379    3    5    0   24  148]
         [   0  812    6    1    1    0]
         ...
        \t- val_loss: 0.536151046634

    Notes that drive the implementation:
      * `val_per_class` and `val_confusion_matrix` are only emitted on
        best-model epochs, so epoch blocks have variable length.
      * `val_loss` comes *after* the multi-line confusion-matrix dump, so a
        naive line-by-line parse would lose it — hence the state machine.
      * A resumed run writes a new log file; epochs are merged across files
        and the later file wins on overlap.
      * The trainer never logs a train accuracy — only `train_loss`.
    """

    LOG_GLOB = 'training_*.log'

    # -- path resolution ---------------------------------------------

    @classmethod
    def resolve_paths(cls, paths: Union[str, Sequence[str]], verbose: bool = True) -> List[str]:
        """Expand files, globs and directories into a sorted list of log files."""
        if isinstance(paths, (str, os.PathLike)):
            paths = [paths]

        resolved: List[str] = []
        for raw in paths:
            path = str(raw)
            if os.path.isdir(path):
                found = sorted(globmod.glob(os.path.join(path, cls.LOG_GLOB)))
                if not found:
                    found = sorted(globmod.glob(os.path.join(path, '*.log')))
                if not found and verbose:
                    print(f"⚠️  No log files in directory: {path}")
                resolved.extend(found)
            elif any(ch in path for ch in '*?['):
                found = sorted(globmod.glob(path))
                if not found and verbose:
                    print(f"⚠️  Glob matched nothing: {path}")
                resolved.extend(found)
            elif os.path.isfile(path):
                resolved.append(path)
            elif verbose:
                print(f"⚠️  Training log not found, skipped: {path}")

        # De-duplicate while keeping order
        seen, unique = set(), []
        for path in resolved:
            key = os.path.abspath(path)
            if key not in seen:
                seen.add(key)
                unique.append(path)
        return unique

    # -- single file --------------------------------------------------

    @classmethod
    def _parse_file(cls, path: str) -> Tuple[Dict[int, Dict[str, float]], Dict[str, str]]:
        epochs: Dict[int, Dict[str, float]] = {}
        meta: Dict[str, str] = {}
        current: Optional[Dict[str, float]] = None
        expect_model_config = False

        with open(path, 'r', encoding='utf-8', errors='replace') as fh:
            for raw_line in fh:
                line = _ANSI_RE.sub('', raw_line.rstrip('\n'))

                # 1) epoch header opens a new record
                epoch_match = _EPOCH_RE.search(line)
                if epoch_match:
                    epoch = int(epoch_match.group(1))
                    current = {'epoch_time_s': float(epoch_match.group(2))}
                    epochs[epoch] = current
                    expect_model_config = False
                    continue

                # 2) "\t- key: value" metric line inside the current record
                metric_match = _METRIC_RE.match(line)
                if metric_match and current is not None:
                    key, value = metric_match.group(1), metric_match.group(2).strip()
                    try:
                        current[key] = float(value)
                    except ValueError:
                        # val_per_class dict / val_confusion_matrix numpy repr
                        pass
                    continue

                # 3) any other timestamped line closes the record and may
                #    carry header metadata
                if _TS_RE.match(line):
                    current = None
                    body = line.split('\t', 1)[1] if '\t' in line else line
                    if body.startswith('Model configuration'):
                        expect_model_config = True
                        continue
                    expect_model_config = False
                    for key, pattern in _HEADER_PATTERNS.items():
                        header_match = pattern.match(body)
                        if header_match:
                            meta[key] = header_match.group(1).strip()
                            break
                    continue

                # 4) the model config is printed on the line after its header
                if expect_model_config and line.strip():
                    meta['model_config'] = line.strip()
                    expect_model_config = False

                # Anything else (confusion-matrix continuation lines,
                # tracebacks, blank lines) is ignored on purpose.

        return epochs, meta

    # -- public entry point -------------------------------------------

    @classmethod
    def parse(
        cls,
        paths: Union[str, Sequence[str]],
        verbose: bool = True,
    ) -> TrainingHistory:
        """
        Parse one or more training logs into a `TrainingHistory`.

        Args:
            paths:   File, glob, directory, or list of any of those.
            verbose: Print per-file diagnostics.

        Returns:
            TrainingHistory with epochs merged across files.
        """
        files = cls.resolve_paths(paths, verbose=verbose)
        if not files:
            raise FileNotFoundError(f"No training log file resolved from: {paths}")

        merged: Dict[int, Dict[str, float]] = {}
        meta: Dict[str, str] = {}
        used_files: List[str] = []
        skipped_files: List[str] = []

        for path in files:
            try:
                epochs, file_meta = cls._parse_file(path)
            except OSError as exc:
                print(f"⚠️  Could not read {path}: {exc}")
                skipped_files.append(path)
                continue

            if not epochs:
                if verbose:
                    print(f"⚠️  No epoch records in {os.path.basename(path)} "
                          f"(aborted/crashed run?) — skipped")
                skipped_files.append(path)
                continue

            meta.update(file_meta)      # later file wins
            merged.update(epochs)       # later file wins on overlapping epochs
            used_files.append(path)
            if verbose:
                print(f"✓ {os.path.basename(path)}: {len(epochs)} epochs "
                      f"({min(epochs)}–{max(epochs)})")

        if not merged:
            raise ValueError(
                f"None of the {len(files)} log file(s) contained a completed epoch. "
                "Training probably crashed before the first epoch ended."
            )

        epoch_numbers = sorted(merged)
        keys = sorted({key for record in merged.values() for key in record})
        metrics = {
            key: np.array([merged[epoch].get(key, np.nan) for epoch in epoch_numbers], dtype=float)
            for key in keys
        }

        history = TrainingHistory(
            epochs=np.array(epoch_numbers, dtype=int),
            metrics=metrics,
            meta=meta,
            files=used_files,
            skipped_files=skipped_files,
        )
        if verbose:
            history.print_summary()
        return history


# ======================================================================
# Training-curve plots (no model / dataset required)
# ======================================================================

def _plot_series(ax, x, history: TrainingHistory, key: str, label: str, **kwargs) -> bool:
    """Plot history[key] on ax; return False if the metric is unavailable."""
    values = history.get(key)
    if values is None:
        return False
    ax.plot(x, values, label=label, **kwargs)
    return True


def _mark_best(ax, history: TrainingHistory, key: str, mode: str, color: str = 'crimson'):
    """Add a marker + annotation on the best epoch of `key`."""
    values = history.get(key)
    if values is None:
        return
    idx = int(np.nanargmin(values) if mode == 'min' else np.nanargmax(values))
    epoch, value = int(history.epochs[idx]), float(values[idx])
    ax.plot(epoch, value, marker='o', markersize=8, markerfacecolor='none',
            markeredgecolor=color, markeredgewidth=2, linestyle='none', zorder=5)
    ax.annotate(
        f"best: {value:.4f}\n@ epoch {epoch}",
        xy=(epoch, value),
        xytext=(8, 12 if mode == 'min' else -28),
        textcoords='offset points',
        fontsize=8, color=color,
        bbox=dict(boxstyle='round,pad=0.3', fc='white', ec=color, alpha=0.85),
    )


def plot_training_curves(
    history: TrainingHistory,
    save_dir: Optional[str] = None,
    plot_format: str = 'pdf',
    show: bool = False,
    dpi: int = 150,
    log_scale: bool = False,
    prefix: str = 'training',
) -> Dict[str, List[str]]:
    """
    Plot the evolution of loss, accuracy, F1, learning rate and confidence
    across epochs, from a parsed training log.

    Produces (one file each):
        <prefix>_loss        train_loss vs val_loss
        <prefix>_accuracy    val accuracy / balanced accuracy / top-5
        <prefix>_f1          val F1 macro / weighted / micro
        <prefix>_lr          learning-rate schedule
        <prefix>_confidence  mean / correct / incorrect confidence
        <prefix>_overview    2x2 summary panel

    Note: the trainer only logs `train_loss` — there is no per-epoch train
    accuracy — so the accuracy plot shows validation metrics only.  This is
    stated on the figure itself rather than silently implied.

    Returns:
        Dict figure-name -> list of written paths.
    """
    x = history.epochs
    written: Dict[str, List[str]] = {}
    title_suffix = f" — {history.meta['experiment']}" if 'experiment' in history.meta else ""

    print(f"\n{'='*66}")
    print("TRAINING CURVES")
    print(f"{'='*66}")

    # ── Loss ────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5.5))
    has_train = _plot_series(ax, x, history, 'train_loss', 'Train loss',
                             color='#1f77b4', lw=1.8)
    has_val = _plot_series(ax, x, history, 'val_loss', 'Validation loss',
                           color='#d62728', lw=1.8)
    if has_train or has_val:
        _mark_best(ax, history, 'val_loss', 'min')
        if log_scale:
            ax.set_yscale('log')
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss' + (' (log scale)' if log_scale else ''), fontsize=12)
        ax.set_title(f"Training and validation loss{title_suffix}",
                     fontsize=13, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        written['loss'] = save_figure(fig, save_dir, f"{prefix}_loss",
                                      plot_format, dpi, show)
    else:
        plt.close(fig)
        print("  ⚠️  no loss metrics in the log — loss plot skipped")

    # ── Accuracy ────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5.5))
    drawn = [
        _plot_series(ax, x, history, 'val_accuracy', 'Val accuracy',
                     color='#2ca02c', lw=1.8),
        _plot_series(ax, x, history, 'val_balanced_accuracy', 'Val balanced accuracy',
                     color='#ff7f0e', lw=1.5),
        _plot_series(ax, x, history, 'val_top_5_accuracy', 'Val top-5 accuracy',
                     color='#9467bd', lw=1.2, linestyle='--', alpha=0.8),
    ]
    if any(drawn):
        _mark_best(ax, history, 'val_accuracy', 'max')
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Accuracy', fontsize=12)
        ax.set_title(f"Validation accuracy evolution{title_suffix}",
                     fontsize=13, fontweight='bold')
        ax.legend(fontsize=10, loc='lower right',
                  title='train accuracy is not logged by the trainer',
                  title_fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        written['accuracy'] = save_figure(fig, save_dir, f"{prefix}_accuracy",
                                          plot_format, dpi, show)
    else:
        plt.close(fig)
        print("  ⚠️  no accuracy metrics in the log — accuracy plot skipped")

    # ── F1 ──────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5.5))
    drawn = [
        _plot_series(ax, x, history, 'val_f1_macro', 'F1 macro', color='#1f77b4', lw=1.8),
        _plot_series(ax, x, history, 'val_f1_weighted', 'F1 weighted', color='#ff7f0e', lw=1.5),
        _plot_series(ax, x, history, 'val_f1_micro', 'F1 micro', color='#2ca02c',
                     lw=1.2, linestyle='--'),
        _plot_series(ax, x, history, 'val_cohen_kappa', "Cohen's kappa", color='#8c564b',
                     lw=1.2, linestyle=':'),
    ]
    if any(drawn):
        _mark_best(ax, history, 'val_f1_macro', 'max')
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title(f"Validation F1 / agreement evolution{title_suffix}",
                     fontsize=13, fontweight='bold')
        ax.legend(fontsize=10, loc='lower right')
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        written['f1'] = save_figure(fig, save_dir, f"{prefix}_f1", plot_format, dpi, show)
    else:
        plt.close(fig)

    # ── Learning rate ───────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 4.5))
    if _plot_series(ax, x, history, 'learning_rate', 'Learning rate',
                    color='#17becf', lw=1.8):
        lr = history.get('learning_rate')
        if np.nanmin(lr) > 0:
            ax.set_yscale('log')
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Learning rate', fontsize=12)
        scheduler = history.meta.get('scheduler', 'schedule')
        ax.set_title(f"Learning-rate schedule ({scheduler}){title_suffix}",
                     fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3, which='both')
        fig.tight_layout()
        written['lr'] = save_figure(fig, save_dir, f"{prefix}_lr", plot_format, dpi, show)
    else:
        plt.close(fig)

    # ── Confidence ──────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))
    drawn = [
        _plot_series(ax, x, history, 'val_mean_confidence', 'Mean confidence',
                     color='#1f77b4', lw=1.8),
        _plot_series(ax, x, history, 'val_mean_correct_confidence',
                     'Mean confidence (correct)', color='#2ca02c', lw=1.4),
        _plot_series(ax, x, history, 'val_mean_incorrect_confidence',
                     'Mean confidence (incorrect)', color='#d62728', lw=1.4),
        _plot_series(ax, x, history, 'val_min_confidence', 'Min confidence',
                     color='#7f7f7f', lw=1.0, linestyle=':'),
    ]
    if any(drawn):
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Confidence', fontsize=12)
        ax.set_title(f"Validation confidence evolution{title_suffix}",
                     fontsize=13, fontweight='bold')
        ax.legend(fontsize=9, loc='best')
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        written['confidence'] = save_figure(fig, save_dir, f"{prefix}_confidence",
                                            plot_format, dpi, show)
    else:
        plt.close(fig)

    # ── Overview 2x2 ────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    ax = axes[0, 0]
    _plot_series(ax, x, history, 'train_loss', 'Train', color='#1f77b4', lw=1.6)
    _plot_series(ax, x, history, 'val_loss', 'Validation', color='#d62728', lw=1.6)
    ax.set_title('Loss', fontweight='bold')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    _plot_series(ax, x, history, 'val_accuracy', 'Accuracy', color='#2ca02c', lw=1.6)
    _plot_series(ax, x, history, 'val_balanced_accuracy', 'Balanced', color='#ff7f0e', lw=1.3)
    ax.set_title('Validation accuracy', fontweight='bold')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Accuracy')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    _plot_series(ax, x, history, 'val_f1_macro', 'F1 macro', color='#1f77b4', lw=1.6)
    _plot_series(ax, x, history, 'val_f1_weighted', 'F1 weighted', color='#ff7f0e', lw=1.3)
    ax.set_title('Validation F1', fontweight='bold')
    ax.set_xlabel('Epoch'); ax.set_ylabel('F1')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    if _plot_series(ax, x, history, 'learning_rate', 'LR', color='#17becf', lw=1.6):
        lr = history.get('learning_rate')
        if lr is not None and np.nanmin(lr) > 0:
            ax.set_yscale('log')
    ax.set_title('Learning rate', fontweight='bold')
    ax.set_xlabel('Epoch'); ax.set_ylabel('LR')
    ax.grid(True, alpha=0.3, which='both')

    n_ep = history.summary['n_epochs']
    fig.suptitle(
        f"Training overview{title_suffix}  |  {n_ep} epochs "
        f"({history.summary['first_epoch']}–{history.summary['last_epoch']})",
        fontsize=14, fontweight='bold'
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    written['overview'] = save_figure(fig, save_dir, f"{prefix}_overview",
                                      plot_format, dpi, show)

    print(f"{'='*66}")
    return written


# ======================================================================
# Inference
# ======================================================================

class Inference:
    """
    The Inference class handles the process of running inference on RSCDMamba model.
    Supports both classification and any other tasks the model performs.

    python -m core.processing.inference \
        --model-path core/processing/output/rscd_mamba_20260526_151512_1gpu/rscd_mamba/checkpoints/best_model.pt \
        --data-path ../RSCD_dataset-1million_reorganized/ \
        --from-checkpoint \
        --evaluate \
        --split separate \
        --plot-all \
        --plot-dir ./plots \
        --export-dir ./plots \
        --train-log core/processing/output/rscd_mamba_20260526_151512_1gpu/rscd_mamba/logs

    Split options:
        --split val      -> validation dataset only (default)
        --split test     -> test dataset only
        --split both     -> val + test concatenated into a single dataset
        --split separate -> val AND test, evaluated and reported independently
                            (equivalent to running the script twice, in one go)
    """

    def __init__(
        self,
        model_path: str,
        img_size: int = 224,
        batch_size: int = 1,
        num_workers: int = 1,
        mean: Optional[List[float]] = None,
        std: Optional[List[float]] = None,
        extensions: Optional[Tuple[str, ...]] = None,
        load_from_checkpoint: bool = False,
        model_instance: Optional[torch.nn.Module] = None,
        device: Optional[torch.device] = None,
        # Supporto per dataset e dataloader esistenti
        dataset_val=None,
        dataloader_val=None,
        # Parametri per build_dataset
        data_set: str = 'RSCD',
        data_path: Optional[str] = None,
        # Scelta del dataset: 'val', 'test', 'both' o 'separate'
        split: str = 'val',
        # Optional explicit class names (default: ImageFolder folder names)
        class_names: Optional[Sequence[str]] = None,
        # Strict state_dict loading — False allows unexpected/missing keys
        # (useful when the checkpoint was saved with a newer model version)
        strict: bool = False,
        # bf16 autocast for every forward pass (evaluation, benchmarks, latency)
        amp: bool = False,
        **kwargs
    ):
        """
        Initializes the Inference class.

        Args:
            model_path: Path to the saved model file or checkpoint file.
            img_size: Input image size.
            batch_size: Batch size for the DataLoader.
            num_workers: Number of workers for the DataLoader.
            mean: Mean for normalization.
            std: Standard deviation for normalization.
            extensions: Supported image file extensions.
            load_from_checkpoint: Whether to load model from checkpoint format.
            model_instance: Optional model instance to use.
            device: Device to use for inference.
            dataset_val: Pre-built validation dataset (bypasses data_path).
            dataloader_val: Pre-built validation dataloader (bypasses data_path).
            data_set: Dataset type ('RSCD', 'IMNET', 'CIFAR', etc.).
            data_path: Path to dataset root directory.
            split: Which dataset split(s) to use: 'val', 'test', 'both' or
                   'separate'.  'separate' loads val and test as two distinct
                   datasets that are evaluated independently.
                   Ignored when dataset_val is provided directly.
            class_names: Optional override for the class names used in reports
                         and plots.  By default the ImageFolder folder names are
                         used (for RSCD: dry, fresh_snow, ice, melted_snow,
                         water, wet).
        """
        if split not in ('val', 'test', 'both', 'separate'):
            raise ValueError(
                f"split must be 'val', 'test', 'both' or 'separate', got '{split}'"
            )

        self.model_path = model_path
        self.load_from_checkpoint = load_from_checkpoint
        self.img_size = img_size
        self.mean = mean or [0.485, 0.456, 0.406]  # ImageNet default
        self.std = std or [0.229, 0.224, 0.225]    # ImageNet default
        self.extensions = extensions or ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
        self.split = split
        self._class_names_override = list(class_names) if class_names else None
        self.amp = amp
        self.strict = strict

        # Gestione device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        # Gestione dataset — self.datasets maps split name -> dataset
        if dataset_val is not None:
            # Dataset fornito direttamente dall'esterno (es. dal training loop)
            self.datasets = {'val': dataset_val}
            self.num_classes = (getattr(dataset_val, 'num_classes', None)
                                or len(dataset_val.classes))
            self.split = 'val'  # forced, il chiamante ha già scelto il dataset
        elif data_path:
            self.datasets, self.num_classes = self._create_datasets(
                data_path,
                data_set,
                img_size,
                split=split,
                **kwargs
            )
        else:
            raise ValueError(
                "Provide either dataset_val or data_path. "
                "Example: Inference(model_path='model.pt', data_path='data/', data_set='RSCD')"
            )

        self.split_names: List[str] = list(self.datasets.keys())

        # Gestione dataloader
        if dataloader_val is not None:
            if len(self.split_names) > 1:
                raise ValueError(
                    "dataloader_val can only be supplied for a single split; "
                    f"got splits {self.split_names}"
                )
            self.dataloaders = {self.split_names[0]: dataloader_val}
            self.batch_size = dataloader_val.batch_size
        else:
            self.batch_size = batch_size
            self.dataloaders = {
                name: DataLoader(
                    dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=num_workers,
                    pin_memory=True if self.device.type == 'cuda' else False
                )
                for name, dataset in self.datasets.items()
            }

        # Carica il modello (una sola volta, condiviso da tutti gli split)
        self.model = self._load_model(instance=model_instance)

        print(f"✓ Inference setup complete")
        print(f"  Device:        {self.device}")
        print(f"  Split mode:    {self.split.upper()}")
        for name in self.split_names:
            print(f"    - {name:<10} {len(self.datasets[name])} samples")
        print(f"  Batch size:    {self.batch_size}")
        print(f"  Num classes:   {self.num_classes}")
        print(f"  Class names:   {', '.join(self._class_names())}")

    # ------------------------------------------------------------------
    # Backward-compatible aliases (single-split callers)
    # ------------------------------------------------------------------

    @property
    def dataset_val(self):
        """The primary dataset — kept for backward compatibility."""
        return self.datasets[self.split_names[0]]

    @property
    def dataloader_val(self):
        """The primary dataloader — kept for backward compatibility."""
        return self.dataloaders[self.split_names[0]]

    # ------------------------------------------------------------------
    # Static / class methods
    # ------------------------------------------------------------------

    @staticmethod
    def get_best_device(force_cpu=False, gpu_id=None):
        if force_cpu:
            print("⚠️ Forcing CPU usage")
            return torch.device("cpu")

        if not torch.cuda.is_available():
            print("⚠️ CUDA not available, falling back to CPU")
            return torch.device("cpu")

        num_gpus = torch.cuda.device_count()

        if gpu_id is not None:
            if gpu_id >= num_gpus:
                raise ValueError(
                    f"GPU {gpu_id} not available. Available GPUs: 0-{num_gpus-1}"
                )
            print(f"✓ Using GPU {gpu_id}: {torch.cuda.get_device_name(gpu_id)}")
            return torch.device(f"cuda:{gpu_id}")

        # AUTO SELECT -> GPU con più memoria libera
        best_gpu = None
        max_free_mem = -1

        for i in range(num_gpus):
            free_mem, total_mem = torch.cuda.mem_get_info(i)
            free_mem_gb = free_mem / 1024**3
            total_mem_gb = total_mem / 1024**3
            print(
                f"GPU {i}: {torch.cuda.get_device_name(i)} | "
                f"Free: {free_mem_gb:.2f} GB / {total_mem_gb:.2f} GB"
            )
            if free_mem > max_free_mem:
                max_free_mem = free_mem
                best_gpu = i

        print(f"✓ Auto-selected GPU {best_gpu}")
        return torch.device(f"cuda:{best_gpu}")

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        data_path: str,
        data_set: str = 'RSCD',
        img_size: int = 224,
        batch_size: int = 1,
        num_workers: int = 1,
        device: Optional[torch.device] = None,
        split: str = 'val',
        **kwargs
    ):
        """
        Create an Inference instance loading model from checkpoint.

        Args:
            checkpoint_path: Path to the checkpoint file.
            data_path: Path to the validation dataset.
            data_set: Dataset type ('RSCD', 'IMNET', etc.).
            img_size: Image size for dataset.
            batch_size: Batch size.
            num_workers: Number of workers.
            device: Device to use.
            split: 'val', 'test', 'both' or 'separate'.

        Returns:
            Inference instance configured with checkpoint.

        Example:
            ```python
            inference = Inference.from_checkpoint(
                checkpoint_path='experiments/best_model.pt',
                data_path='data/',
                data_set='RSCD',
                img_size=224,
                split='separate'
            )
            results_by_split = inference.evaluate_splits()
            ```
        """
        return cls(
            model_path=checkpoint_path,
            data_path=data_path,
            data_set=data_set,
            img_size=img_size,
            batch_size=batch_size,
            num_workers=num_workers,
            load_from_checkpoint=True,
            device=device,
            split=split,
            **kwargs
        )

    @classmethod
    def from_training_config(
        cls,
        checkpoint_path: str,
        dataset_val,
        batch_size: int = 1,
        num_workers: int = 1,
        device: Optional[torch.device] = None
    ):
        """
        Create Inference from existing training configuration.

        Args:
            checkpoint_path: Path to checkpoint.
            dataset_val: Validation dataset from training.
            batch_size: Batch size for inference.
            num_workers: Number of workers.
            device: Device to use.

        Returns:
            Inference instance.

        Example:
            ```python
            # Durante il training
            dataset_val, num_classes = build_dataset(is_train=False, is_test=False, args=args)

            # Per inference
            inference = Inference.from_training_config(
                checkpoint_path='best_model.pt',
                dataset_val=dataset_val
            )
            ```
        """
        return cls(
            model_path=checkpoint_path,
            dataset_val=dataset_val,
            batch_size=batch_size,
            num_workers=num_workers,
            load_from_checkpoint=True,
            device=device
        )

    # ------------------------------------------------------------------
    # Dataset creation
    # ------------------------------------------------------------------

    def _create_datasets(
        self,
        data_path: str,
        data_set: str,
        img_size: int,
        split: str = 'val',
        **kwargs
    ) -> Tuple[Dict[str, Any], int]:
        """
        Creates the dataset(s) using build_dataset.

        Args:
            data_path: Path to dataset root.
            data_set: Dataset type.
            img_size: Image size.
            split: 'val', 'test', 'both' or 'separate'.
                   - 'val'      -> {'val': ds}                 (is_train=False, is_test=False)
                   - 'test'     -> {'test': ds}                (is_train=False, is_test=True)
                   - 'both'     -> {'both': ConcatDataset}     val + test concatenated
                   - 'separate' -> {'val': ds, 'test': ds}     evaluated independently

        Returns:
            Tuple of (dict split-name -> dataset, num_classes).
        """
        # Crea un oggetto args-like per build_dataset
        class Args:
            pass

        args = Args()
        args.data_path = data_path
        args.data_set = data_set
        args.img_size = img_size
        args.eval_crop_ratio = kwargs.get('eval_crop_ratio', 0.875)
        args.color_jitter = kwargs.get('color_jitter', None)
        args.aa = kwargs.get('aa', 'rand-m9-mstd0.5-inc1')
        args.train_interpolation = kwargs.get('train_interpolation', 'bicubic')
        args.reprob = kwargs.get('reprob', 0.25)
        args.remode = kwargs.get('remode', 'pixel')
        args.recount = kwargs.get('recount', 1)

        if data_set.startswith('INAT'):
            args.inat_category = kwargs.get('inat_category', 'name')

        def _build(is_test: bool, label: str):
            dataset, num_classes = build_dataset(
                is_train=False, is_test=is_test, args=args
            )
            print(f"✓ {label} dataset created: {data_set} | "
                  f"{len(dataset)} samples | {num_classes} classes")
            return dataset, num_classes

        try:
            if split == 'val':
                dataset, num_classes = _build(False, 'Validation')
                return {'val': dataset}, num_classes

            if split == 'test':
                dataset, num_classes = _build(True, 'Test')
                return {'test': dataset}, num_classes

            if split in ('both', 'separate'):
                dataset_val, num_classes = _build(False, 'Validation')
                dataset_test, num_classes_test = _build(True, 'Test')

                if num_classes != num_classes_test:
                    raise RuntimeError(
                        f"Val and test datasets have different num_classes: "
                        f"{num_classes} vs {num_classes_test}"
                    )

                if split == 'separate':
                    print(f"✓ Separate evaluation mode: val ({len(dataset_val)}) and "
                          f"test ({len(dataset_test)}) will be reported independently")
                    return {'val': dataset_val, 'test': dataset_test}, num_classes

                dataset = ConcatDataset([dataset_val, dataset_test])

                # ConcatDataset non ha .classes; lo propaga dal primo dataset
                if hasattr(dataset_val, 'classes'):
                    dataset.classes = dataset_val.classes
                if hasattr(dataset_val, 'num_classes'):
                    dataset.num_classes = dataset_val.num_classes

                print(
                    f"✓ Combined (val+test) dataset created: {data_set} | "
                    f"{len(dataset_val)} val + {len(dataset_test)} test = "
                    f"{len(dataset)} total samples | {num_classes} classes"
                )
                return {'both': dataset}, num_classes

            raise ValueError(
                f"Unknown split '{split}'. Use 'val', 'test', 'both' or 'separate'."
            )

        except Exception as e:
            raise RuntimeError(f"Failed to create dataset (split='{split}'): {e}")

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_model(
        self,
        instance: Optional[torch.nn.Module] = None
    ) -> torch.nn.Module:
        """
        Loads the PyTorch model from the specified path.

        Args:
            instance: Optional model instance class.

        Returns:
            Loaded model ready for inference.
        """
        try:
            if self.load_from_checkpoint:
                checkpoint_loader = CheckpointLoader(device=self.device)
                model_cls = instance if instance is not None else RSCDMamba

                model = checkpoint_loader.load_model_for_inference(
                    checkpoint_path=self.model_path,
                    model_instance=model_cls,
                    strict=self.strict
                )

                if model is None:
                    raise RuntimeError(f"CheckpointLoader returned None")

                print(f"✓ Model loaded from checkpoint: {self.model_path}")
            else:
                model = torch.load(
                    self.model_path,
                    weights_only=False,
                    map_location=self.device
                )
                print(f"✓ Model loaded from file: {self.model_path}")

            model.eval()
            model = model.to(self.device)

            device_check = next(model.parameters()).device
            print(f"✓ Model is on device: {device_check}")

            if self.device.type == 'cuda' and not device_check.type == 'cuda':
                raise RuntimeError(f"Model should be on CUDA but is on {device_check}")

            return model

        except Exception as e:
            raise RuntimeError(f"Failed to load model from {self.model_path}: {e}")

    # ------------------------------------------------------------------
    # Complexity / benchmark
    # ------------------------------------------------------------------
    #
    # Every analysis section keeps the lines it printed under a `_report`
    # key: `_json_safe` strips it from the JSON and the TXT export writes
    # exactly what was shown on the terminal.  The JSON schema is shared
    # with the JIT-side inference.py, whose stacking run reads
    # model_summary_mamba.json / single_inference_mamba.json back to build
    # the full-pipeline complexity and latency.

    MODEL_NAME = 'mamba'
    FLOPS_CONVENTION = 'FLOPs = 2 x MACs'

    @staticmethod
    def _emit(lines: List[str]) -> List[str]:
        for line in lines:
            print(line)
        return lines

    @staticmethod
    def _timing_stats(times_ms: Sequence[float]) -> Dict[str, float]:
        t = np.asarray(times_ms, dtype=np.float64)
        return {
            'mean_ms': float(np.mean(t)),
            'std_ms': float(np.std(t)),
            'median_ms': float(np.median(t)),
            'min_ms': float(np.min(t)),
            'max_ms': float(np.max(t)),
            'p95_ms': float(np.percentile(t, 95)),
            'p99_ms': float(np.percentile(t, 99)),
        }

    @staticmethod
    def _format_count(num: Optional[float]) -> str:
        if num is None:
            return 'N/A'
        if num >= 1e9:
            return f"{num/1e9:.2f}G"
        if num >= 1e6:
            return f"{num/1e6:.2f}M"
        if num >= 1e3:
            return f"{num/1e3:.2f}K"
        return str(num)

    def _autocast(self):
        """bf16 autocast (as in training, core/processing/training.py) when --amp."""
        return torch.autocast(device_type='cuda', dtype=torch.bfloat16,
                              enabled=self.amp and self.device.type == 'cuda')

    @property
    def _precision(self) -> str:
        return 'amp (bf16 autocast)' if self.amp and self.device.type == 'cuda' else 'fp32'

    def _sync(self):
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)

    def _device_name(self) -> str:
        if self.device.type == 'cuda':
            return torch.cuda.get_device_name(self.device)
        import platform
        return platform.processor() or platform.machine() or 'CPU'

    def _fvcore_macs(self, input_size: Tuple[int, int, int, int]) -> Tuple[int, Dict[str, int]]:
        """
        MACs through fvcore, with handlers for the ops FlopCounterMode misses.

        RSCDMamba runs two custom CUDA kernels wrapped in autograd Functions
        (the selective scan and, in eval mode, the fused 11x11 depthwise
        conv) plus torch.fft in EinFFT.  torch's FlopCounterMode sees none of
        them.  fvcore traces the graph, so custom handlers can be attached to
        the `prim::PythonOp` nodes: the selective-scan formula is the one
        already used by `RSCDMamba.flops()`; the depthwise conv is counted
        as B*C*H*W*kh*kw.  Ops still without a handler (FFT) are returned so
        they can be reported.
        """
        from fvcore.nn import FlopCountAnalysis
        from fvcore.nn.jit_handles import get_shape
        from core.utility.selective_scan import selective_scan_state_flop_jit

        def dwconv_flop_jit(inputs, outputs):
            batch, channels, height, width = get_shape(outputs[0])
            kh, kw = get_shape(inputs[1])[-2:]
            return batch * channels * height * width * kh * kw

        handles = {
            # element-wise ops ignored, as in RSCDMamba.flops()
            'aten::silu': None,
            'aten::neg': None,
            'aten::exp': None,
            'aten::flip': None,
            'prim::PythonOp.SelectiveScanStateFn': selective_scan_state_flop_jit,
            'prim::PythonOp.DepthwiseFunction': dwconv_flop_jit,
        }
        dummy_input = torch.randn(*input_size, device=self.device)
        self.model.eval()
        with torch.no_grad():
            analysis = FlopCountAnalysis(self.model, (dummy_input,)).set_op_handle(**handles)
            analysis.unsupported_ops_warnings(False)
            analysis.uncalled_modules_warnings(False)
            total = int(analysis.total())
            unsupported = dict(analysis.unsupported_ops())
        return total, unsupported

    def calculate_model_complexity(
        self,
        input_size: Optional[Tuple[int, int, int, int]] = None,
        verbose: bool = True
    ) -> Dict:
        """
        Model complexity: parameter counts, size in MB, FLOPs and MACs.

        FLOPs follow the JIT-side convention (FLOPs = 2 x MACs) so that the two
        models can be summed.  Primary method: fvcore with custom handlers
        (see `_fvcore_macs`).  Fallback: torch FlopCounterMode, which misses
        the custom kernels, so its value is flagged as an underestimate.
        """
        if self.model is None:
            raise RuntimeError("Model not loaded.")
        input_size = input_size or (1, 3, self.img_size, self.img_size)
        fmt = self._format_count

        params = list(self.model.parameters())
        total_params = sum(p.numel() for p in params)
        trainable_params = sum(p.numel() for p in params if p.requires_grad)
        non_trainable_params = total_params - trainable_params

        param_size = sum(p.nelement() * p.element_size() for p in params)
        buffer_size = sum(b.nelement() * b.element_size() for b in self.model.buffers())
        model_size_mb = (param_size + buffer_size) / 1024 / 1024
        checkpoint_size_mb = (os.path.getsize(self.model_path) / 1024**2
                              if self.model_path and os.path.isfile(self.model_path)
                              else None)

        results = {
            'total_params': total_params,
            'trainable_params': trainable_params,
            'non_trainable_params': non_trainable_params,
            'total_params_formatted': fmt(total_params),
            'trainable_params_formatted': fmt(trainable_params),
            'model_size_mb': model_size_mb,
            'checkpoint_size_mb': checkpoint_size_mb,
            'flops': None,
            'flops_formatted': 'N/A',
            'macs': None,
            'macs_formatted': 'N/A',
            'gflops': None,
            'gmacs': None,
            'flops_convention': self.FLOPS_CONVENTION,
            'flops_warning': None,
            'unsupported_ops': None,
            'flops_torch_counter': None,
            'method': 'manual',
            'input_size': list(input_size),
            'img_size': self.img_size,
            'device': str(self.device),
            'model_type': type(self.model).__name__,
            'torch_version': torch.__version__,
        }

        def _set_macs(macs: int, method: str):
            results.update({
                'flops': 2 * macs,
                'flops_formatted': fmt(2 * macs),
                'macs': macs,
                'macs_formatted': fmt(macs),
                'gflops': 2 * macs / 1e9,
                'gmacs': macs / 1e9,
                'method': method,
            })

        fvcore_error = None
        try:
            macs, unsupported = self._fvcore_macs(input_size)
            _set_macs(macs, 'fvcore_custom_handles')
            results['unsupported_ops'] = unsupported or None
            if unsupported:
                results['flops_warning'] = (
                    f"ops without a FLOP handler, not counted: "
                    f"{', '.join(sorted(unsupported))}")
        except Exception as exc:
            fvcore_error = str(exc)
            results['fvcore_error'] = fvcore_error

        # FlopCounterMode is kept for reference; alone it underestimates.
        if FLOP_COUNTER_AVAILABLE:
            try:
                dummy_input = torch.randn(*input_size, device=self.device)
                self.model.eval()
                flop_counter = FlopCounterMode(display=False)
                # Grad mode must stay ON: under no_grad / inference_mode the
                # counter's module tracker fails ("Expected gradient function
                # to be set") and the FLOPs silently end up N/A.  Only a
                # forward pass runs, nothing is back-propagated.
                with torch.enable_grad(), flop_counter:
                    _ = self.model(dummy_input)
                torch_flops = int(flop_counter.get_total_flops())
                results['flops_torch_counter'] = torch_flops
                if results['flops'] is None:
                    _set_macs(torch_flops // 2, 'torch_flop_counter')
                    results['flops_warning'] = (
                        "FlopCounterMode cannot see the custom CUDA kernels "
                        "(selective scan, depthwise conv) nor torch.fft: "
                        "the value is an UNDERESTIMATE")
            except Exception as exc:
                if verbose:
                    print(f"⚠️  FlopCounterMode failed: {exc}")

        if results['flops'] is None and verbose:
            print(f"⚠️  Could not calculate FLOPs: {fvcore_error}")

        size_ckpt = (f"{checkpoint_size_mb:.2f} MB" if checkpoint_size_mb is not None
                     else 'N/A')
        lines = [
            "=" * 60,
            "MODEL COMPLEXITY ANALYSIS",
            "=" * 60,
            f"Total Parameters:      {results['total_params_formatted']} ({total_params:,})",
            f"Trainable Parameters:  {results['trainable_params_formatted']} ({trainable_params:,})",
            f"Non-trainable Params:  {fmt(non_trainable_params)} ({non_trainable_params:,})",
            f"Model Size (memory):   {model_size_mb:.2f} MB",
            f"Checkpoint on disk:    {size_ckpt}",
            f"FLOPs:                 {results['flops_formatted']}"
            + (f"  ({results['gflops']:.3f} GFLOPs)" if results['gflops'] is not None else ''),
            f"MACs:                  {results['macs_formatted']}"
            + (f"  ({results['gmacs']:.3f} GMACs)" if results['gmacs'] is not None else ''),
            f"Convention:            {self.FLOPS_CONVENTION}",
            f"Method:                {results['method']}",
        ]
        if results['flops_torch_counter'] is not None and results['method'] != 'torch_flop_counter':
            lines.append(f"FlopCounterMode only:  {fmt(results['flops_torch_counter'])} FLOPs "
                         f"(misses the custom kernels)")
        lines += [
            f"Input Size:            {tuple(input_size)}",
            f"Device:                {self.device}",
            f"Model Type:            {results['model_type']}",
        ]
        if results['flops_warning']:
            lines.append(f"⚠️  {results['flops_warning']}")
        lines.append("=" * 60)
        results['_report'] = self._emit(lines) if verbose else lines
        return results

    def benchmark_inference(
        self,
        input_size: Optional[Tuple[int, int, int, int]] = None,
        num_runs: int = 100,
        warmup_runs: int = 10
    ) -> Dict:
        """Forward-pass latency on one synthetic input (same autocast as --amp)."""
        if self.model is None:
            raise RuntimeError("Model not loaded.")
        input_size = input_size or (1, 3, self.img_size, self.img_size)
        dummy_input = torch.randn(*input_size).to(self.device)
        self.model.eval()

        print(f"Warming up with {warmup_runs} runs...")
        with torch.no_grad():
            for _ in range(warmup_runs):
                with self._autocast():
                    _ = self.model(dummy_input)
        self._sync()

        print(f"Running {num_runs} inference benchmarks...")
        times = []
        with torch.no_grad():
            for i in range(num_runs):
                start_time = time.perf_counter()
                with self._autocast():
                    _ = self.model(dummy_input)
                self._sync()
                times.append((time.perf_counter() - start_time) * 1000)
                if (i + 1) % 20 == 0:
                    print(f"Completed {i + 1}/{num_runs} runs")

        times = np.array(times)
        results = {
            'mean_time_ms': float(np.mean(times)),
            'std_time_ms': float(np.std(times)),
            'min_time_ms': float(np.min(times)),
            'max_time_ms': float(np.max(times)),
            'median_time_ms': float(np.median(times)),
            'p95_time_ms': float(np.percentile(times, 95)),
            'p99_time_ms': float(np.percentile(times, 99)),
            'fps': 1000.0 / float(np.mean(times)),
            'throughput': (input_size[0] * 1000.0) / float(np.mean(times)),
            'input_size': list(input_size),
            'device': str(self.device),
            'device_name': self._device_name(),
            'precision': self._precision,
            'num_runs': num_runs,
            'warmup_runs': warmup_runs,
        }

        lines = [
            "=" * 60,
            "INFERENCE BENCHMARK RESULTS (synthetic input, forward only)",
            "=" * 60,
            f"Mean Time:    {results['mean_time_ms']:.2f} ± {results['std_time_ms']:.2f} ms",
            f"Median Time:  {results['median_time_ms']:.2f} ms",
            f"P95 / P99:    {results['p95_time_ms']:.2f} / {results['p99_time_ms']:.2f} ms",
            f"Min/Max Time: {results['min_time_ms']:.2f} / {results['max_time_ms']:.2f} ms",
            f"FPS (batch={input_size[0]}): {results['fps']:.2f}",
            f"Throughput:   {results['throughput']:.2f} images/sec",
            f"Device:       {self.device} ({results['device_name']})",
            f"Precision:    {self._precision}",
            "=" * 60,
        ]
        print()
        results['_report'] = self._emit(lines)
        return results

    def memory_benchmark(
        self,
        input_size: Optional[Tuple[int, int, int, int]] = None,
        num_runs: int = 10
    ) -> Dict:
        """
        Benchmark memory usage during inference.

        NOTES on CUDA memory tracking:
          - torch.cuda.memory_allocated() returns memory used by live tensors.
            After the model is loaded the baseline is non-zero, so it is
            subtracted and the deltas represent inference overhead only.
          - memory_reserved() is the total pool size claimed from the OS;
            it is never smaller than memory_allocated().
          - memory_efficiency = allocated / reserved * 100. When both are 0
            we report 'N/A' instead of NaN.
        """
        if self.model is None:
            raise RuntimeError("Model not loaded.")
        input_size = input_size or (1, 3, self.img_size, self.img_size)
        process = None

        if self.device.type == 'cuda':
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            initial_allocated = torch.cuda.memory_allocated() / 1024**2
            initial_reserved = torch.cuda.memory_reserved() / 1024**2
            torch.cuda.reset_peak_memory_stats()
            print("Starting GPU memory benchmark...")
            print(f"  Baseline allocated (model loaded): {initial_allocated:.2f} MB")
            print(f"  Baseline reserved  (pool size):    {initial_reserved:.2f} MB")
        else:
            process = psutil.Process(os.getpid())
            initial_allocated = process.memory_info().rss / 1024**2
            initial_reserved = None
            print("Starting CPU memory benchmark...")
            print(f"  Initial RSS: {initial_allocated:.2f} MB")

        dummy_input = torch.randn(*input_size).to(self.device)
        allocated_samples, reserved_samples, peak_samples = [], [], []

        print(f"Running {num_runs} memory profiling runs...")
        with torch.no_grad():
            for i in range(num_runs):
                if self.device.type == 'cuda':
                    torch.cuda.reset_peak_memory_stats()

                with self._autocast():
                    _ = self.model(dummy_input)

                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
                    allocated_samples.append(torch.cuda.memory_allocated() / 1024**2)
                    reserved_samples.append(torch.cuda.memory_reserved() / 1024**2)
                    peak_samples.append(torch.cuda.max_memory_allocated() / 1024**2)
                else:
                    allocated_samples.append(process.memory_info().rss / 1024**2)

                if (i + 1) % 5 == 0:
                    print(f"  Completed {i + 1}/{num_runs} runs")

        if self.device.type == 'cuda':
            avg_allocated = float(np.mean(allocated_samples))
            avg_reserved = float(np.mean(reserved_samples))
            max_peak = float(np.max(peak_samples))

            if avg_reserved > 0:
                efficiency = (avg_allocated / avg_reserved) * 100.0
                efficiency_str = f"{efficiency:.1f}%"
            else:
                efficiency = float('nan')
                efficiency_str = "N/A (reserved=0, pool may be pre-allocated)"

            results = {
                'device': 'GPU (CUDA)',
                'device_name': self._device_name(),
                'precision': self._precision,
                'input_size': list(input_size),
                'num_runs': num_runs,
                'baseline_allocated_mb': initial_allocated,
                'baseline_reserved_mb': initial_reserved,
                'avg_allocated_mb': avg_allocated,
                'max_allocated_mb': float(np.max(allocated_samples)),
                'avg_reserved_mb': avg_reserved,
                'max_reserved_mb': float(np.max(reserved_samples)),
                'avg_peak_mb': float(np.mean(peak_samples)),
                'max_peak_mb': max_peak,
                'inference_overhead_mb': max_peak - initial_allocated,
                'memory_efficiency': efficiency,
                'memory_efficiency_str': efficiency_str,
            }
            lines = [
                "=" * 60,
                "GPU MEMORY USAGE RESULTS",
                "=" * 60,
                f"Baseline (model loaded):   {initial_allocated:.2f} MB allocated  /  "
                f"{initial_reserved:.2f} MB reserved",
                "During inference:",
                f"  Avg allocated:           {avg_allocated:.2f} MB",
                f"  Peak allocated:          {results['max_allocated_mb']:.2f} MB",
                f"  Avg reserved (pool):     {avg_reserved:.2f} MB",
                f"  Peak during inference:   {max_peak:.2f} MB",
                f"Inference overhead:        {results['inference_overhead_mb']:.2f} MB  "
                f"(peak - baseline)",
                f"Memory efficiency:         {efficiency_str}  (allocated / reserved)",
                f"Precision:                 {self._precision}",
                "=" * 60,
            ]
        else:
            avg_rss = float(np.mean(allocated_samples))
            max_rss = float(np.max(allocated_samples))
            results = {
                'device': 'CPU',
                'device_name': self._device_name(),
                'precision': self._precision,
                'input_size': list(input_size),
                'num_runs': num_runs,
                'baseline_allocated_mb': initial_allocated,
                'avg_rss_mb': avg_rss,
                'max_rss_mb': max_rss,
                'inference_overhead_mb': max_rss - initial_allocated,
            }
            lines = [
                "=" * 60,
                "CPU MEMORY USAGE RESULTS",
                "=" * 60,
                f"Initial RSS:    {initial_allocated:.2f} MB",
                f"Average RSS:    {avg_rss:.2f} MB",
                f"Peak RSS:       {max_rss:.2f} MB",
                f"Overhead:       {results['inference_overhead_mb']:.2f} MB",
                "=" * 60,
            ]

        print()
        results['_report'] = self._emit(lines)
        return results

    def model_summary(
        self,
        input_size: Optional[Tuple[int, int, int, int]] = None,
        include_benchmark: bool = False,
        include_memory: bool = False,
        benchmark_runs: int = 50,
        memory_runs: int = 10
    ) -> Dict:
        """Complexity, plus optionally the latency and memory benchmarks."""
        print("🔍 Starting Model Analysis...")
        print("=" * 60)

        results = {
            'complexity': self.calculate_model_complexity(input_size, verbose=True),
            'benchmark': None,
            'memory': None,
        }

        if include_benchmark:
            print("\n" + "=" * 60)
            results['benchmark'] = self.benchmark_inference(
                input_size, benchmark_runs, warmup_runs=10)

        if include_memory:
            print("\n" + "=" * 60)
            results['memory'] = self.memory_benchmark(input_size, memory_runs)

        print("\n" + "=" * 60)
        print("✅ Model Analysis Complete!")
        print("=" * 60)
        return results

    # ------------------------------------------------------------------
    # Single-image inference latency
    # ------------------------------------------------------------------

    _PHASE_LABELS = {
        'load_ms': 'Load + preprocess',
        'h2d_ms': 'Host -> device',
        'forward_ms': 'Forward pass',
        'post_ms': 'Post-process',
        'model_ms': 'Model (no load)',
        'total_ms': 'Total per image',
    }

    def single_image_inference(
        self,
        num_samples: int = 50,
        warmup_runs: int = 10,
        seed: Optional[int] = 42,
        split_name: Optional[str] = None,
    ) -> Dict:
        """
        Latency of one prediction at a time (batch = 1) on real images.

        Unlike `benchmark_inference` (one synthetic tensor, forward only) this
        times the whole path of a single prediction, split into phases:
            load_ms     dataset[idx]: image decode + eval transforms (CPU)
            h2d_ms      host -> device copy
            forward_ms  model forward pass (same autocast as --amp)
            post_ms     softmax + argmax + copy of the result to the host
            model_ms    h2d + forward + post: latency once the image is
                        already decoded and preprocessed
            total_ms    load + model_ms: end-to-end latency of one image
        The first `warmup_runs` forwards (CUDA context, cuDNN autotuning,
        allocator growth) are excluded; without them a run on few images
        mostly measures the cold start.
        """
        if self.model is None:
            raise RuntimeError("Model not loaded.")
        split_name = split_name or ('test' if 'test' in self.datasets
                                    else self.split_names[0])
        dataset = self.datasets[split_name]
        class_names = self._class_names(split_name)
        rng = np.random.default_rng(seed)
        indices = rng.choice(len(dataset), size=min(num_samples, len(dataset)),
                             replace=False)
        on_cuda = self.device.type == 'cuda'
        self.model.eval()

        print(f"\nSingle-image inference: {len(indices)} images from "
              f"{split_name.upper()} (batch=1, warm-up {warmup_runs} runs, "
              f"{self._precision})")

        with torch.no_grad():
            warm_input = dataset[int(indices[0])][0].unsqueeze(0).to(self.device)
            for _ in range(warmup_runs):
                with self._autocast():
                    _ = self.model(warm_input)
            self._sync()
            del warm_input

        process = None
        if on_cuda:
            baseline_mb = torch.cuda.memory_allocated(self.device) / 1024**2
            torch.cuda.reset_peak_memory_stats(self.device)
        else:
            process = psutil.Process(os.getpid())
            baseline_mb = process.memory_info().rss / 1024**2
        peak_rss_mb = baseline_mb

        phases: Dict[str, List[float]] = {
            k: [] for k in ('load_ms', 'h2d_ms', 'forward_ms', 'post_ms',
                            'model_ms', 'total_ms')}
        rows: List[Dict[str, Any]] = []

        with torch.no_grad():
            for idx in indices:
                idx = int(idx)
                t0 = time.perf_counter()
                image, target = dataset[idx]
                t1 = time.perf_counter()
                x = image.unsqueeze(0).to(self.device)
                self._sync()
                t2 = time.perf_counter()
                with self._autocast():
                    outputs = self.model(x)
                if isinstance(outputs, (tuple, list)):
                    outputs = outputs[0]
                self._sync()
                t3 = time.perf_counter()
                probs = torch.softmax(outputs.float(), dim=1)
                confidence, prediction = probs.max(dim=1)
                confidence, prediction = float(confidence.item()), int(prediction.item())
                t4 = time.perf_counter()

                if process is not None:
                    peak_rss_mb = max(peak_rss_mb, process.memory_info().rss / 1024**2)

                times = {
                    'load_ms': (t1 - t0) * 1000,
                    'h2d_ms': (t2 - t1) * 1000,
                    'forward_ms': (t3 - t2) * 1000,
                    'post_ms': (t4 - t3) * 1000,
                    'model_ms': (t4 - t1) * 1000,
                    'total_ms': (t4 - t0) * 1000,
                }
                for key, value in times.items():
                    phases[key].append(value)

                target = int(target)
                rows.append({
                    'index': idx,
                    'target': target,
                    'target_name': class_names[target],
                    'prediction': prediction,
                    'prediction_name': class_names[prediction],
                    'correct': int(prediction == target),
                    'confidence': confidence,
                    **times,
                })

        peak_mb = (torch.cuda.max_memory_allocated(self.device) / 1024**2
                   if on_cuda else peak_rss_mb)
        memory = {
            'kind': 'cuda_allocated' if on_cuda else 'cpu_rss',
            'baseline_mb': baseline_mb,
            'peak_mb': peak_mb,
            'overhead_mb': peak_mb - baseline_mb,
        }

        timing = {k: self._timing_stats(v) for k, v in phases.items()}
        latency = timing['total_ms']['mean_ms']
        summary = {
            'split': split_name,
            'num_samples': len(rows),
            'warmup_runs': warmup_runs,
            'seed': seed,
            'batch_size': 1,
            'device': str(self.device),
            'device_name': self._device_name(),
            'precision': self._precision,
            'img_size': self.img_size,
            'latency_ms': latency,
            'model_latency_ms': timing['model_ms']['mean_ms'],
            'forward_latency_ms': timing['forward_ms']['mean_ms'],
            'throughput_imgs_per_sec': 1000.0 / latency if latency > 0 else None,
            'sample_accuracy': float(np.mean([r['correct'] for r in rows])),
            'timing': timing,
            'memory': memory,
            '_rows': rows,
        }

        lines = [
            "=" * 60,
            "SINGLE-IMAGE INFERENCE (batch = 1)",
            "=" * 60,
            f"Model:      {self.MODEL_NAME}",
            f"Split:      {split_name}",
            f"Images:     {len(rows)}  (warm-up {warmup_runs} runs excluded, seed {seed})",
            f"Device:     {summary['device']} ({summary['device_name']})",
            f"Precision:  {self._precision}",
            "",
            f"{'Phase':<20}{'mean':>9}{'std':>9}{'median':>9}{'p95':>9}{'max':>9}   [ms]",
        ]
        for key, stats in timing.items():
            lines.append(
                f"{self._PHASE_LABELS.get(key, key):<20}{stats['mean_ms']:>9.3f}"
                f"{stats['std_ms']:>9.3f}{stats['median_ms']:>9.3f}"
                f"{stats['p95_ms']:>9.3f}{stats['max_ms']:>9.3f}")
        lines += [
            "",
            f"Latency per image:  {latency:.3f} ms  ->  "
            f"{summary['throughput_imgs_per_sec']:.2f} images/sec",
            f"Memory ({memory['kind']}):  peak {peak_mb:.2f} MB, "
            f"overhead {memory['overhead_mb']:.2f} MB",
            f"Accuracy on these images: {summary['sample_accuracy']:.4f} (sanity check only)",
            "=" * 60,
        ]
        summary['_report'] = self._emit(lines)
        return summary

    # ------------------------------------------------------------------
    # Complexity / latency exports
    # ------------------------------------------------------------------

    def _write_report(self, sections: Sequence[Optional[Dict]], title: str,
                      path: str) -> str:
        with open(path, 'w', encoding='utf-8') as fh:
            fh.write(f"{title}\nModel: {self.MODEL_NAME}\n"
                     f"Checkpoint: {self.model_path}\n"
                     f"Generated: {_dt.datetime.now().isoformat(timespec='seconds')}\n\n")
            for section in sections:
                if section and section.get('_report'):
                    fh.write("\n".join(section['_report']) + "\n\n")
        print(f"  ✓ saved → {path}")
        return path

    def _payload(self, kind: str, summary: Dict) -> Dict:
        payload = {'kind': kind, 'model': self.MODEL_NAME,
                   'checkpoint': self.model_path,
                   'generated': _dt.datetime.now().isoformat(timespec='seconds')}
        payload.update(summary)
        return payload

    def export_model_summary(self, summary: Dict, out_dir: str) -> Dict[str, str]:
        """
        Write model_summary_mamba.json / .txt.

        The JIT-side stacking run reads the JSON back (same schema as
        model_summary_jit.json) to sum the complexity of the two base models.
        """
        os.makedirs(out_dir, exist_ok=True)
        print(f"\nExporting model analysis [{self.MODEL_NAME}] → {out_dir}")
        return {
            'json': self._write_json(
                self._payload('model_summary', summary),
                os.path.join(out_dir, f"model_summary_{self.MODEL_NAME}.json")),
            'txt': self._write_report(
                [summary.get(k) for k in ('complexity', 'benchmark', 'memory')],
                "MODEL ANALYSIS",
                os.path.join(out_dir, f"model_summary_{self.MODEL_NAME}.txt")),
        }

    def export_single_inference(self, summary: Dict, out_dir: str) -> Dict[str, str]:
        """Write single_inference_mamba.json / .txt / .csv (read back by the stacking)."""
        os.makedirs(out_dir, exist_ok=True)
        print(f"\nExporting single-image inference [{self.MODEL_NAME}] → {out_dir}")
        written = {
            'json': self._write_json(
                self._payload('single_inference', summary),
                os.path.join(out_dir, f"single_inference_{self.MODEL_NAME}.json")),
            'txt': self._write_report(
                [summary], "SINGLE-IMAGE INFERENCE",
                os.path.join(out_dir, f"single_inference_{self.MODEL_NAME}.txt")),
        }
        rows = summary.get('_rows') or []
        if rows:
            path = os.path.join(out_dir, f"single_inference_{self.MODEL_NAME}.csv")
            with open(path, 'w', newline='', encoding='utf-8') as fh:
                writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                for row in rows:
                    writer.writerow({k: (f"{v:.6f}" if isinstance(v, float) else v)
                                     for k, v in row.items()})
            print(f"  ✓ saved → {path}")
            written['csv'] = path
        return written

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _class_names(self, split: Optional[str] = None) -> List[str]:
        """
        Class names used in reports and plots.

        Order: explicit override -> dataset `.classes` (ImageFolder folder names,
        alphabetical) -> generic `class_i`.
        """
        if self._class_names_override:
            return list(self._class_names_override)

        key = split if split in self.datasets else self.split_names[0]
        names = getattr(self.datasets[key], 'classes', None)
        if names:
            return list(names)
        return [f"class_{i}" for i in range(self.num_classes)]

    def tensor_to_image(self, tensor: torch.Tensor) -> np.ndarray:
        """
        Converts a tensor to numpy image for visualization.

        Args:
            tensor: Input tensor.

        Returns:
            Numpy array image in [0, 1] range.
        """
        img = tensor.cpu().detach().numpy()

        if img.ndim == 4:
            img = img.squeeze(0)

        if img.shape[0] in [1, 3]:
            img = np.transpose(img, (1, 2, 0))

        if img.shape[-1] == 1:
            img = np.squeeze(img, axis=-1)

        should_denormalize = (
            self.mean is not None and
            self.std is not None and
            not (np.array(self.mean) == 0).all() and
            not (np.array(self.std) == 255).all()
        )

        if should_denormalize:
            mean = np.array(self.mean)
            std = np.array(self.std)

            if img.ndim == 3 and img.shape[-1] == 3:
                img = img * std + mean
            elif img.ndim == 2:
                img = img * std[0] + mean[0]
            else:
                img = img * std + mean

        return np.clip(img, 0, 1)

    def _inference_step(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Single inference step."""
        input_tensor = input_tensor.to(self.device)

        if not next(self.model.parameters()).is_cuda and self.device.type == 'cuda':
            print(f"WARNING: Model not on CUDA! Moving it now...")
            self.model = self.model.to(self.device)

        with torch.no_grad():
            with self._autocast():
                outputs = self.model(input_tensor)
        return outputs.float()

    # ------------------------------------------------------------------
    # Extended metrics
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_calibration(
        confidences: np.ndarray,
        correct: np.ndarray,
        n_bins: int = 15
    ) -> Dict[str, Any]:
        """
        Confidence-calibration statistics.

        Expected Calibration Error (ECE) is the sample-weighted mean gap between
        bin accuracy and bin confidence; Maximum Calibration Error (MCE) is the
        largest such gap over non-empty bins.  A perfectly calibrated model has
        ECE = 0 (accuracy inside a bin equals the confidence in that bin).

        Args:
            confidences: Predicted probability of the predicted class, shape (N,).
            correct:     Boolean array, True where the prediction was right.
            n_bins:      Number of equal-width confidence bins.

        Returns:
            Dict with 'ece', 'mce', 'n_bins' and the per-bin arrays.
        """
        confidences = np.asarray(confidences, dtype=float)
        correct = np.asarray(correct, dtype=bool)
        n_samples = len(confidences)

        edges = np.linspace(0.0, 1.0, n_bins + 1)
        bin_acc = np.full(n_bins, np.nan)
        bin_conf = np.full(n_bins, np.nan)
        bin_count = np.zeros(n_bins, dtype=int)

        ece = 0.0
        mce = 0.0
        signed = 0.0
        for i in range(n_bins):
            lower, upper = edges[i], edges[i + 1]
            # Include the right edge in the last bin so conf == 1.0 is counted
            mask = ((confidences > lower) & (confidences <= upper)) if i > 0 else \
                   ((confidences >= lower) & (confidences <= upper))
            count = int(mask.sum())
            bin_count[i] = count
            if count == 0:
                continue
            acc = float(correct[mask].mean())
            conf = float(confidences[mask].mean())
            bin_acc[i] = acc
            bin_conf[i] = conf
            gap = abs(acc - conf)
            ece += (count / n_samples) * gap
            mce = max(mce, gap)
            signed += (count / n_samples) * (conf - acc)

        return {
            'ece': float(ece),
            'mce': float(mce),
            # Signed counterpart of ECE: > 0 overconfident, < 0 underconfident
            'signed_gap': float(signed),
            'n_bins': n_bins,
            'bin_edges': edges,
            'bin_accuracy': bin_acc,
            'bin_confidence': bin_conf,
            'bin_count': bin_count,
        }

    def _extended_metrics(
        self,
        targets: np.ndarray,
        predictions: np.ndarray,
        probabilities: np.ndarray,
        confidences: np.ndarray,
        class_names: List[str],
    ) -> Dict[str, Any]:
        """
        Compute the full metric set for one split.

        Includes macro/micro/weighted averages, Cohen's kappa, Matthews
        correlation coefficient, top-k accuracy, per-class precision / recall /
        F1 / support / specificity / accuracy / mean confidence, and calibration.

        Note: the training loop reports `matthews_corrcoef` as 0.0 for
        multi-class problems (core/processing/loss.py only computes it for
        binary); here it is computed properly.
        """
        labels = list(range(self.num_classes))
        cm = confusion_matrix(targets, predictions, labels=labels)
        correct_mask = (predictions == targets)

        precision_pc = precision_score(targets, predictions, labels=labels,
                                       average=None, zero_division=0)
        recall_pc = recall_score(targets, predictions, labels=labels,
                                 average=None, zero_division=0)
        f1_pc = f1_score(targets, predictions, labels=labels,
                         average=None, zero_division=0)

        total = int(cm.sum())
        per_class: Dict[str, Dict[str, float]] = {}
        for i, name in enumerate(class_names[:self.num_classes]):
            tp = int(cm[i, i])
            support = int(cm[i, :].sum())
            predicted = int(cm[:, i].sum())
            fp = predicted - tp
            fn = support - tp
            tn = total - tp - fp - fn
            class_mask = (targets == i)
            per_class[name] = {
                'precision': float(precision_pc[i]),
                'recall': float(recall_pc[i]),
                'f1': float(f1_pc[i]),
                'support': support,
                'predicted_count': predicted,
                'true_positives': tp,
                'false_positives': fp,
                'false_negatives': fn,
                'specificity': float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0,
                'accuracy': float(tp / support) if support > 0 else 0.0,
                'mean_confidence': float(confidences[class_mask].mean()) if class_mask.any() else 0.0,
            }

        metrics: Dict[str, Any] = {
            'accuracy': float(accuracy_score(targets, predictions)),
            'error_rate': float(1.0 - accuracy_score(targets, predictions)),
            'balanced_accuracy': float(balanced_accuracy_score(targets, predictions)),
            'cohen_kappa': float(cohen_kappa_score(targets, predictions)),
            'matthews_corrcoef': float(matthews_corrcoef(targets, predictions)),
            'per_class': per_class,
            'confusion_matrix': cm.tolist(),
            '_confusion_matrix': cm,
        }

        for average in ('macro', 'micro', 'weighted'):
            metrics[f'precision_{average}'] = float(
                precision_score(targets, predictions, labels=labels,
                                average=average, zero_division=0))
            metrics[f'recall_{average}'] = float(
                recall_score(targets, predictions, labels=labels,
                             average=average, zero_division=0))
            metrics[f'f1_{average}'] = float(
                f1_score(targets, predictions, labels=labels,
                         average=average, zero_division=0))

        # Primary precision/recall/f1 kept at the top level for backward
        # compatibility: macro for multi-class, binary for 2 classes.
        if self.num_classes > 2:
            metrics['precision'] = metrics['precision_macro']
            metrics['recall'] = metrics['recall_macro']
            metrics['f1_score'] = metrics['f1_macro']
        else:
            metrics['precision'] = float(precision_score(targets, predictions,
                                                         average='binary', zero_division=0))
            metrics['recall'] = float(recall_score(targets, predictions,
                                                   average='binary', zero_division=0))
            metrics['f1_score'] = float(f1_score(targets, predictions,
                                                  average='binary', zero_division=0))

        # Top-k accuracy needs the full probability matrix
        if probabilities is not None and len(probabilities):
            for k in range(1, min(5, self.num_classes) + 1):
                try:
                    metrics[f'top_{k}_accuracy'] = float(
                        top_k_accuracy_score(targets, probabilities, k=k, labels=labels))
                except ValueError as exc:
                    print(f"⚠️  top-{k} accuracy unavailable: {exc}")

            # ROC-AUC / average precision live here (not only in plot_curves)
            # so they always reach the JSON, the per-class CSV and the split
            # comparison; '_curves' is reused by plot_curves().
            curves = compute_curve_data(targets, probabilities, self.num_classes)
            if curves is not None:
                names = class_names[:self.num_classes]
                if curves['skipped']:
                    print("⚠️  ROC/PR undefined (no positive or no negative sample) for: "
                          + ', '.join(names[i] for i in curves['skipped']))
                auc_scores = auc_scores_from_curves(curves, names)
                for name in names:
                    per_class[name]['roc_auc'] = auc_scores[f'roc_auc_{name}']
                    per_class[name]['ap'] = auc_scores[f'pr_auc_{name}']
                metrics['auc_scores'] = auc_scores
                metrics['_curves'] = curves

        # Confidence + calibration
        metrics['mean_confidence'] = float(np.mean(confidences))
        metrics['median_confidence'] = float(np.median(confidences))
        metrics['min_confidence'] = float(np.min(confidences))
        metrics['max_confidence'] = float(np.max(confidences))
        metrics['mean_correct_confidence'] = (
            float(confidences[correct_mask].mean()) if correct_mask.any() else 0.0)
        metrics['mean_incorrect_confidence'] = (
            float(confidences[~correct_mask].mean()) if (~correct_mask).any() else 0.0)

        calibration = self._compute_calibration(confidences, correct_mask)
        metrics['calibration'] = {
            'ece': calibration['ece'],
            'mce': calibration['mce'],
            'signed_gap': calibration['signed_gap'],
            'n_bins': calibration['n_bins'],
        }
        metrics['_calibration'] = calibration

        return metrics

    # ------------------------------------------------------------------
    # Public inference / evaluation methods
    # ------------------------------------------------------------------

    def inference(
        self,
        images_number: int = 50,
        seed: Optional[int] = 42,
        warmup_runs: int = 10,
        split_name: Optional[str] = None,
    ) -> Dict:
        """
        Single-image (batch = 1) inference latency on `images_number` random
        images.  Kept as the historical entry point; see
        `single_image_inference` for what is measured.
        """
        return self.single_image_inference(
            num_samples=images_number,
            warmup_runs=warmup_runs,
            seed=seed,
            split_name=split_name,
        )

    def evaluate_dataset(
        self,
        batch_size: Optional[int] = None,
        verbose: bool = True,
        save_predictions: bool = False,
        output_file: Optional[str] = None,
        track_performance: bool = True,
        split: Optional[str] = None,
    ) -> Dict:
        """
        Evaluate the model on one loaded split.

        Args:
            batch_size: Batch size for evaluation (default: same as dataloader).
            verbose: Print progress and results.
            save_predictions: Keep predictions/targets/confidences in the result
                              dict (and write them if output_file is given).
            output_file: File path for saving the metrics as JSON.
            track_performance: Track time and memory usage.
            split: Which loaded split to evaluate. Defaults to the primary one.
                   Use `evaluate_splits()` to evaluate every loaded split.

        Returns:
            Dictionary with evaluation metrics and performance data.
        """
        split_name = split or self.split_names[0]
        if split_name not in self.datasets:
            raise ValueError(
                f"Split '{split_name}' is not loaded. Available: {self.split_names}"
            )

        results = self._evaluate_one(
            split_name,
            batch_size=batch_size,
            verbose=verbose,
            save_predictions=save_predictions,
            track_performance=track_performance,
        )

        if output_file:
            self._write_json(results, output_file)

        return results

    def evaluate_splits(
        self,
        batch_size: Optional[int] = None,
        verbose: bool = True,
        save_predictions: bool = False,
        track_performance: bool = True,
    ) -> Dict[str, Dict]:
        """
        Evaluate every loaded split independently, reusing the single loaded
        model (the checkpoint is never read twice).

        With `split='separate'` this yields the same numbers you would get by
        running the script once with `--split val` and once with `--split test`,
        but in one process and with a side-by-side comparison at the end.

        Returns:
            Dict split-name -> result dict.
        """
        results_by_split: Dict[str, Dict] = {}
        for name in self.split_names:
            if verbose:
                print(f"\n{'#'*66}")
                print(f"#  EVALUATING SPLIT: {name.upper()}")
                print(f"{'#'*66}")
            results_by_split[name] = self._evaluate_one(
                name,
                batch_size=batch_size,
                verbose=verbose,
                save_predictions=save_predictions,
                track_performance=track_performance,
            )

        if len(results_by_split) > 1 and verbose:
            self.print_split_comparison(results_by_split)

        return results_by_split

    def _evaluate_one(
        self,
        split_name: str,
        batch_size: Optional[int] = None,
        verbose: bool = True,
        save_predictions: bool = False,
        track_performance: bool = True,
    ) -> Dict:
        """Run the forward pass over one split and compute every metric."""
        dataset = self.datasets[split_name]
        base_loader = self.dataloaders[split_name]

        if batch_size is None:
            batch_size = base_loader.batch_size

        if batch_size != base_loader.batch_size:
            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=base_loader.num_workers,
                pin_memory=True if self.device.type == 'cuda' else False
            )
        else:
            dataloader = base_loader

        all_predictions   = []
        all_targets       = []
        all_confidences   = []
        all_probabilities = []   # shape: (N, num_classes) — needed for ROC/PR curves
        batch_times       = []

        total_images = len(dataset)
        processed_images = 0

        if track_performance:
            if self.device.type == 'cuda':
                # Synchronize so all pending ops are done, then read the
                # current footprint (model weights already live in the pool).
                # Do NOT call empty_cache() here — that would evict the model
                # weights and give a falsely-low baseline.
                torch.cuda.synchronize()
                # Baseline: memory actually allocated by live tensors (model weights)
                initial_allocated_mb = torch.cuda.memory_allocated() / 1024**2
                # Reserved = total pool size PyTorch holds from the OS
                initial_reserved_mb  = torch.cuda.memory_reserved()  / 1024**2
                # Reset peak counter AFTER reading baseline so peak reflects
                # inference overhead on top of the already-loaded model
                torch.cuda.reset_peak_memory_stats()
                memory_samples = []
            else:
                process = psutil.Process(os.getpid())
                initial_allocated_mb = process.memory_info().rss / 1024**2
                initial_reserved_mb  = None
                memory_samples = []

        if verbose:
            print(f"Starting evaluation on {total_images} images (split={split_name.upper()})...")
            print(f"Batch size: {batch_size}")
            if track_performance and self.device.type == 'cuda':
                print(f"GPU memory baseline — allocated: {initial_allocated_mb:.1f} MB  "
                      f"reserved: {initial_reserved_mb:.1f} MB")
        start_time = time.time()

        self.model.eval()
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(dataloader):
                if track_performance:
                    batch_start = time.perf_counter()

                outputs = self._inference_step(inputs)

                if track_performance:
                    if self.device.type == 'cuda':
                        torch.cuda.synchronize()
                    batch_times.append((time.perf_counter() - batch_start) * 1000)
                    memory_samples.append(
                        torch.cuda.memory_allocated() / 1024**2
                        if self.device.type == 'cuda'
                        else process.memory_info().rss / 1024**2
                    )

                if outputs.dim() > 1:
                    probs_batch = torch.softmax(outputs, dim=1)
                    preds = torch.argmax(probs_batch, dim=1)
                    confs = probs_batch.max(dim=1)[0]
                    all_probabilities.append(probs_batch.cpu().numpy())
                else:
                    preds = outputs
                    confs = torch.ones_like(preds)

                all_predictions.extend(preds.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                all_confidences.extend(confs.cpu().numpy())

                processed_images += len(inputs)

                if verbose and (batch_idx + 1) % max(1, len(dataloader) // 10) == 0:
                    elapsed_time = time.time() - start_time
                    progress = processed_images / total_images * 100
                    eta = elapsed_time / processed_images * (total_images - processed_images)
                    throughput = processed_images / elapsed_time
                    print(f"Progress: {progress:.1f}% ({processed_images}/{total_images}) - "
                          f"ETA: {eta:.1f}s - Throughput: {throughput:.2f} imgs/s")

        all_predictions   = np.array(all_predictions)
        all_targets       = np.array(all_targets)
        all_confidences   = np.array(all_confidences)
        all_probabilities = np.vstack(all_probabilities) if all_probabilities else np.array([])

        class_names = self._class_names(split_name)

        # The dataloader is built with shuffle=False, so the running position
        # is the dataset index — this is what lets us fetch the misclassified
        # images back later.
        all_indices = np.arange(len(all_targets))

        metrics = self._extended_metrics(
            all_targets, all_predictions, all_probabilities,
            all_confidences, class_names
        )

        # Performance metrics
        performance_metrics = None
        if track_performance:
            total_time = time.time() - start_time
            t_arr = np.array(batch_times)
            m_arr = np.array(memory_samples)
            mean_time_per_image = float(np.mean(t_arr)) / batch_size

            if self.device.type == 'cuda':
                # Peak allocated AFTER baseline reset = pure inference overhead
                peak_allocated_mb = torch.cuda.max_memory_allocated() / 1024**2
                peak_reserved_mb  = torch.cuda.max_memory_reserved()  / 1024**2
                performance_metrics = {
                    'device': 'GPU (CUDA)',
                    'total_time_s': total_time,
                    'total_images': total_images,
                    'batch_size': batch_size,
                    'num_batches': len(batch_times),
                    'mean_batch_time_ms': float(np.mean(t_arr)),
                    'std_batch_time_ms':  float(np.std(t_arr)),
                    'median_batch_time_ms': float(np.median(t_arr)),
                    'mean_time_per_image_ms': mean_time_per_image,
                    'throughput_imgs_per_sec': total_images / total_time,
                    # Baseline = model weights already on GPU before the loop
                    'baseline_allocated_mb': initial_allocated_mb,
                    'baseline_reserved_mb':  initial_reserved_mb,
                    # Peak values captured during the inference loop
                    'peak_allocated_mb': peak_allocated_mb,
                    'peak_reserved_mb':  peak_reserved_mb,
                    'avg_allocated_mb':  float(np.mean(m_arr)),
                    # Overhead = extra memory needed on top of the loaded model
                    'inference_overhead_mb': peak_allocated_mb - initial_allocated_mb,
                    'memory_per_image_mb': max(peak_allocated_mb - initial_allocated_mb, 0) / total_images,
                }
            else:
                peak_rss_mb = float(np.max(m_arr))
                performance_metrics = {
                    'device': 'CPU',
                    'total_time_s': total_time,
                    'total_images': total_images,
                    'batch_size': batch_size,
                    'num_batches': len(batch_times),
                    'mean_batch_time_ms': float(np.mean(t_arr)),
                    'std_batch_time_ms':  float(np.std(t_arr)),
                    'median_batch_time_ms': float(np.median(t_arr)),
                    'mean_time_per_image_ms': mean_time_per_image,
                    'throughput_imgs_per_sec': total_images / total_time,
                    'baseline_allocated_mb': initial_allocated_mb,
                    'peak_allocated_mb': peak_rss_mb,
                    'avg_allocated_mb':  float(np.mean(m_arr)),
                    'inference_overhead_mb': peak_rss_mb - initial_allocated_mb,
                    'memory_per_image_mb': max(peak_rss_mb - initial_allocated_mb, 0) / total_images,
                }

        results: Dict[str, Any] = {
            'split': split_name,
            'split_mode': self.split,
            'model_path': self.model_path,
            'n_samples': len(all_targets),
            'num_classes': self.num_classes,
            'class_names': class_names,
            'performance': performance_metrics,
            # Arrays kept in memory for the plots; '_'-prefixed keys are
            # stripped before JSON serialisation.
            '_probabilities': all_probabilities,
            '_targets': all_targets,
            '_predictions': all_predictions,
            '_confidences': all_confidences,
            '_indices': all_indices,
            'predictions': all_predictions.tolist() if save_predictions else None,
            'targets': all_targets.tolist() if save_predictions else None,
            'confidences': all_confidences.tolist() if save_predictions else None,
        }
        results.update(metrics)

        if verbose:
            self._print_evaluation_results(results)

        return results

    def _print_evaluation_results(self, results: Dict):
        """Print the full metric report for one split."""
        split_name = results['split']
        performance = results.get('performance')
        class_names = results['class_names']

        print(f"\n{'='*66}")
        print(f"DATASET EVALUATION RESULTS  [split={split_name.upper()}]")
        print(f"{'='*66}")
        print(f"Total Images: {results['n_samples']}")

        if performance:
            print(f"Processing Time: {performance['total_time_s']:.2f}s "
                  f"({performance['mean_time_per_image_ms']:.2f}ms per image)")
            print(f"Throughput: {performance['throughput_imgs_per_sec']:.2f} images/sec")

        print(f"\n--- Overall ---")
        print(f"Accuracy:          {results['accuracy']:.4f}")
        print(f"Error rate:        {results['error_rate']:.4f}")
        print(f"Balanced Accuracy: {results['balanced_accuracy']:.4f}")
        print(f"Cohen's kappa:     {results['cohen_kappa']:.4f}")
        print(f"Matthews corrcoef: {results['matthews_corrcoef']:.4f}")

        print(f"\n--- Averaged (macro / micro / weighted) ---")
        print(f"{'':<12}{'macro':>10}{'micro':>10}{'weighted':>10}")
        for metric in ('precision', 'recall', 'f1'):
            row = ''.join(f"{results[f'{metric}_{avg}']:>10.4f}"
                          for avg in ('macro', 'micro', 'weighted'))
            print(f"{metric:<12}{row}")

        topk = sorted(k for k in results if re.fullmatch(r'top_\d+_accuracy', k))
        if topk:
            print(f"\n--- Top-k accuracy ---")
            for key in topk:
                print(f"{key:<18} {results[key]:.4f}")

        auc_scores = results.get('auc_scores')
        if auc_scores:
            print(f"\n--- ROC-AUC / AP (macro / micro / weighted) ---")
            print(f"{'':<12}{'macro':>10}{'micro':>10}{'weighted':>10}")
            for kind, label in (('roc', 'ROC-AUC'), ('pr', 'AP')):
                row = ''.join(f"{auc_scores[f'{kind}_auc_{avg}']:>10.4f}"
                              for avg in ('macro', 'micro', 'weighted'))
                print(f"{label:<12}{row}")

        print(f"\n--- Confidence ---")
        print(f"Mean:               {results['mean_confidence']:.4f}")
        print(f"Median:             {results['median_confidence']:.4f}")
        print(f"Min / Max:          {results['min_confidence']:.4f} / {results['max_confidence']:.4f}")
        print(f"Mean (correct):     {results['mean_correct_confidence']:.4f}")
        print(f"Mean (incorrect):   {results['mean_incorrect_confidence']:.4f}")
        calibration = results['calibration']
        print(f"ECE ({calibration['n_bins']} bins):       {calibration['ece']:.4f}")
        print(f"MCE:                {calibration['mce']:.4f}")
        tendency = 'overconfident' if calibration['signed_gap'] > 0 else 'underconfident'
        print(f"Signed gap:         {calibration['signed_gap']:+.4f}  ({tendency})")

        print(f"\n--- Per class ---")
        header = (f"  {'class':<16}{'prec':>8}{'recall':>8}{'f1':>8}"
                  f"{'spec':>8}{'support':>9}{'pred':>8}{'conf':>8}")
        print(header)
        print(f"  {'-'*(len(header)-2)}")
        for name in class_names:
            stats = results['per_class'].get(name)
            if stats is None:
                continue
            print(f"  {name:<16}{stats['precision']:>8.4f}{stats['recall']:>8.4f}"
                  f"{stats['f1']:>8.4f}{stats['specificity']:>8.4f}"
                  f"{stats['support']:>9}{stats['predicted_count']:>8}"
                  f"{stats['mean_confidence']:>8.4f}")

        print("\nDetailed Classification Report:")
        print(classification_report(
            results['_targets'],
            results['_predictions'],
            labels=list(range(self.num_classes)),
            target_names=class_names,
            zero_division=0
        ))

        print("Confusion matrix (rows = true, cols = predicted):")
        cm = results['_confusion_matrix']
        width = max(8, max(len(n) for n in class_names) + 1)
        print(' ' * width + ''.join(f"{n[:7]:>8}" for n in class_names))
        for i, name in enumerate(class_names):
            print(f"{name:<{width}}" + ''.join(f"{int(v):>8}" for v in cm[i]))

        if performance:
            print(f"\n{'='*66}")
            print("PERFORMANCE METRICS")
            print(f"{'='*66}")
            print(f"Device:      {performance['device']}")
            print(f"Batch Size:  {performance['batch_size']}")
            print(f"Num Batches: {performance['num_batches']}")
            print(f"\nTiming:")
            print(f"  Mean Batch Time:     {performance['mean_batch_time_ms']:.2f} ± "
                  f"{performance['std_batch_time_ms']:.2f} ms")
            print(f"  Median Batch Time:   {performance['median_batch_time_ms']:.2f} ms")
            print(f"  Mean Time per Image: {performance['mean_time_per_image_ms']:.2f} ms")
            print(f"  Total Throughput:    {performance['throughput_imgs_per_sec']:.2f} imgs/s")
            print(f"\nMemory:")
            print(f"  Baseline allocated:  {performance['baseline_allocated_mb']:.2f} MB  "
                  f"(model weights on GPU)")
            if performance.get('baseline_reserved_mb') is not None:
                print(f"  Baseline reserved:   {performance['baseline_reserved_mb']:.2f} MB  "
                      f"(total CUDA pool)")
            print(f"  Peak allocated:      {performance['peak_allocated_mb']:.2f} MB")
            if performance.get('peak_reserved_mb') is not None:
                print(f"  Peak reserved:       {performance['peak_reserved_mb']:.2f} MB")
            print(f"  Avg allocated:       {performance['avg_allocated_mb']:.2f} MB")
            print(f"  Inference overhead:  {performance['inference_overhead_mb']:.2f} MB  "
                  f"(peak − baseline)")
            print(f"  Per image:           {performance['memory_per_image_mb']:.4f} MB")

        print(f"{'='*66}")

    # ------------------------------------------------------------------
    # Split comparison
    # ------------------------------------------------------------------

    COMPARISON_METRICS = [
        ('n_samples', 'Samples', '{:,}'),
        ('accuracy', 'Accuracy', '{:.4f}'),
        ('error_rate', 'Error rate', '{:.4f}'),
        ('balanced_accuracy', 'Balanced accuracy', '{:.4f}'),
        ('precision_macro', 'Precision (macro)', '{:.4f}'),
        ('recall_macro', 'Recall (macro)', '{:.4f}'),
        ('f1_macro', 'F1 (macro)', '{:.4f}'),
        ('f1_weighted', 'F1 (weighted)', '{:.4f}'),
        ('cohen_kappa', "Cohen's kappa", '{:.4f}'),
        ('matthews_corrcoef', 'Matthews corrcoef', '{:.4f}'),
        ('top_5_accuracy', 'Top-5 accuracy', '{:.4f}'),
        ('mean_confidence', 'Mean confidence', '{:.4f}'),
        ('mean_correct_confidence', 'Confidence (correct)', '{:.4f}'),
        ('mean_incorrect_confidence', 'Confidence (incorrect)', '{:.4f}'),
    ]

    def print_split_comparison(
        self,
        results_by_split: Dict[str, Dict],
        out_dir: Optional[str] = None,
    ) -> Optional[str]:
        """
        Print a side-by-side table of the splits, and optionally write it to CSV.

        Returns:
            Path of the written CSV, or None.
        """
        splits = list(results_by_split.keys())
        rows: List[Tuple[str, List[str]]] = []

        for key, label, fmt in self.COMPARISON_METRICS:
            values = []
            present = False
            for split in splits:
                value = results_by_split[split].get(key)
                if value is None:
                    values.append('—')
                else:
                    present = True
                    values.append(fmt.format(value))
            if present:
                rows.append((label, values))

        # AUC rows (present whenever probabilities were available)
        for key, label in (('roc_auc_macro', 'ROC-AUC (macro)'),
                           ('roc_auc_micro', 'ROC-AUC (micro)'),
                           ('roc_auc_weighted', 'ROC-AUC (weighted)'),
                           ('pr_auc_macro', 'PR-AUC (macro)'),
                           ('pr_auc_micro', 'PR-AUC (micro)'),
                           ('pr_auc_weighted', 'PR-AUC (weighted)')):
            values, present = [], False
            for split in splits:
                auc_scores = results_by_split[split].get('auc_scores') or {}
                value = auc_scores.get(key)
                if value is None:
                    values.append('—')
                else:
                    present = True
                    values.append(f"{value:.4f}")
            if present:
                rows.append((label, values))

        # Calibration rows
        for key, label, fmt in (('ece', 'ECE', '{:.4f}'),
                                ('mce', 'MCE', '{:.4f}'),
                                ('signed_gap', 'Signed gap', '{:+.4f}')):
            values, present = [], False
            for split in splits:
                calibration = results_by_split[split].get('calibration') or {}
                value = calibration.get(key)
                if value is None:
                    values.append('—')
                else:
                    present = True
                    values.append(fmt.format(value))
            if present:
                rows.append((label, values))

        label_width = max(len(label) for label, _ in rows) + 2
        col_width = 14

        print(f"\n{'='*66}")
        print("SPLIT COMPARISON")
        print(f"{'='*66}")
        print(f"{'Metric':<{label_width}}" + ''.join(f"{s.upper():>{col_width}}" for s in splits))
        print('-' * (label_width + col_width * len(splits)))
        for label, values in rows:
            print(f"{label:<{label_width}}" + ''.join(f"{v:>{col_width}}" for v in values))
        print(f"{'='*66}")

        if not out_dir:
            return None

        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"comparison_{'_vs_'.join(splits)}.csv")
        with open(path, 'w', newline='', encoding='utf-8') as fh:
            writer = csv.writer(fh)
            writer.writerow(['metric'] + splits)
            for label, values in rows:
                writer.writerow([label] + values)
        print(f"  ✓ saved → {path}")
        return path

    # ------------------------------------------------------------------
    # Curve plots — ROC and Precision-Recall
    # ------------------------------------------------------------------

    def plot_curves(
        self,
        results: Dict,
        save_dir: Optional[str] = None,
        show: bool = False,
        dpi: int = 150,
        plot_format: str = 'pdf',
    ) -> Dict[str, float]:
        """
        Plot and optionally save the multiclass AUC-ROC and Precision-Recall
        curves: one-vs-rest curve of every class plus the micro- and
        macro-average curves.  The PR figure also shows iso-F1 curves.
        When the model is good enough for the curves to hug a corner
        (macro AUC >= 0.95, macro AP >= 0.9) the corner is magnified in an inset.
        With two classes only the positive-class curve is drawn.

        Args:
            results:     Dict returned by evaluate_dataset().  Must contain the
                         '_probabilities' and '_targets' numpy arrays that are
                         stored there automatically.
            save_dir:    Directory where the two files are saved.
                         If None the figures are not saved to disk.
            show:        Call plt.show() after building each figure.
            dpi:         Resolution of saved raster figures (PNG only).
            plot_format: 'pdf' (default), 'png' or 'both'.

        Returns:
            Dict with per-class and macro / micro / weighted AUC scores, e.g.:
            {
              'roc_auc_dry': 0.999,
              'roc_auc_macro': 0.998,
              'roc_auc_micro': 0.999,
              'pr_auc_dry': 0.997,
              'pr_auc_macro': 0.996,
              ...
            }
            The same dict is also stored in `results['auc_scores']`.
        """
        probs   = results.get('_probabilities')
        targets = results.get('_targets')

        if probs is None or len(probs) == 0:
            raise ValueError(
                "results dict does not contain '_probabilities'. "
                "Run evaluate_dataset() first — it stores probabilities automatically."
            )

        split       = results.get('split', 'split')
        class_names = results.get('class_names') or self._class_names(split)

        # Curves are normally computed once in _extended_metrics()
        curves = results.get('_curves')
        if curves is None:
            curves = compute_curve_data(targets, probs, self.num_classes)
            if curves is None:
                raise ValueError("no class has both positive and negative samples")
            if curves['skipped']:
                print("⚠️  ROC/PR undefined (no positive or no negative sample) for: "
                      + ', '.join(class_names[i] for i in curves['skipped']))
        auc_scores = auc_scores_from_curves(curves, class_names[:self.num_classes])

        binary  = curves['macro'] is None
        classes = [i for i in curves['valid'] if not binary or i == 1] or curves['valid'][:1]

        # ── Colour palette ──────────────────────────────────────────────
        cmap   = plt.get_cmap('tab10')
        colors = [cmap(i % 10) for i in range(self.num_classes)]

        # ── Figure 1: AUC-ROC ───────────────────────────────────────────
        fig_roc, ax_roc = plt.subplots(figsize=(8, 7))
        roc_lines = []

        for i in classes:
            fpr, tpr, roc_auc = curves['roc'][i]
            style = dict(color=colors[i], lw=1.8)
            ax_roc.plot(fpr, tpr, label=f"{class_names[i]}  (AUC = {roc_auc:.4f})", **style)
            roc_lines.append((fpr, tpr, style))

        if binary:
            title = f"AUC = {auc_scores['roc_auc_macro']:.4f}"
        else:
            for avg, style in (('micro', MICRO_STYLE), ('macro', MACRO_STYLE)):
                fpr, tpr, roc_auc = curves[avg]['roc']
                ax_roc.plot(fpr, tpr, label=f"{avg.capitalize()} avg  (AUC = {roc_auc:.4f})",
                            **style)
                roc_lines.append((fpr, tpr, style))
            title = (f"Micro AUC = {auc_scores['roc_auc_micro']:.4f}  |  "
                     f"Macro AUC = {auc_scores['roc_auc_macro']:.4f}")

        ax_roc.plot([0, 1], [0, 1], 'k:', lw=1, alpha=0.5, label="Chance level  (AUC = 0.5)")

        if auc_scores['roc_auc_macro'] >= 0.95:
            # Top-left corner; the lower bound follows the worst curve
            x_max = 0.05
            y_min = min(_value_at(0.1 * x_max, x, y) for x, y, _ in roc_lines)
            _add_zoom_inset(ax_roc, roc_lines, (0, x_max), (y_min, 1.0),
                            [0.35, 0.35, 0.4, 0.4])

        ax_roc.set_xlim([-0.01, 1.01])
        ax_roc.set_ylim([-0.01, 1.02])
        ax_roc.set_xlabel("False Positive Rate", fontsize=12)
        ax_roc.set_ylabel("True Positive Rate", fontsize=12)
        ax_roc.set_title(f"ROC Curves — {split.upper()}\n{title}",
                         fontsize=13, fontweight='bold')
        ax_roc.legend(loc="lower right", fontsize=9)
        ax_roc.grid(True, alpha=0.3)
        fig_roc.tight_layout()
        save_figure(fig_roc, save_dir, f"roc_curves_{split}", plot_format, dpi, show)

        # ── Figure 2: AUC-Precision-Recall ──────────────────────────────
        fig_pr, ax_pr = plt.subplots(figsize=(8, 7))
        _draw_iso_f1(ax_pr)
        pr_lines = []

        for i in classes:
            rec, prec, ap = curves['pr'][i]
            style = dict(color=colors[i], lw=1.8, drawstyle='steps-post')
            ax_pr.plot(rec, prec, label=f"{class_names[i]}  (AP = {ap:.4f})", **style)
            pr_lines.append((rec, prec, style))

        if binary:
            title = f"AP = {auc_scores['pr_auc_macro']:.4f}"
        else:
            rec, prec, ap = curves['micro']['pr']
            style = dict(MICRO_STYLE, drawstyle='steps-post')
            ax_pr.plot(rec, prec, label=f"Micro avg  (AP = {ap:.4f})", **style)
            pr_lines.append((rec, prec, style))
            rec, prec, ap = curves['macro']['pr']
            ax_pr.plot(rec, prec, label=f"Macro avg  (AP = {ap:.4f})", **MACRO_STYLE)
            pr_lines.append((rec, prec, MACRO_STYLE))
            title = (f"Micro AP = {auc_scores['pr_auc_micro']:.4f}  |  "
                     f"Macro AP = {auc_scores['pr_auc_macro']:.4f}")

        if auc_scores['pr_auc_macro'] >= 0.9:
            # Top-right corner
            y_min = min(_value_at(0.9, x, y) for x, y, _ in pr_lines)
            _add_zoom_inset(ax_pr, pr_lines, (0.8, 1.0), (y_min, 1.0),
                            [0.3, 0.35, 0.4, 0.4])

        ax_pr.set_xlim([-0.01, 1.01])
        ax_pr.set_ylim([-0.01, 1.02])
        ax_pr.set_xlabel("Recall", fontsize=12)
        ax_pr.set_ylabel("Precision", fontsize=12)
        ax_pr.set_title(f"Precision-Recall Curves — {split.upper()}\n{title}",
                        fontsize=13, fontweight='bold')
        ax_pr.legend(loc="lower left", fontsize=9)
        ax_pr.grid(True, alpha=0.3)
        fig_pr.tight_layout()
        save_figure(fig_pr, save_dir, f"pr_curves_{split}", plot_format, dpi, show)

        # ── Summary print ────────────────────────────────────────────────
        print(f"\n{'='*60}")
        print(f"AUC SCORES  [split={split.upper()}]")
        print(f"{'='*60}")
        col = 22
        print(f"  {'Class':<{col}} {'ROC-AUC':>10}   {'PR-AUC (AP)':>12}")
        print(f"  {'-'*col} {'-'*10}   {'-'*12}")
        for cls_name in class_names:
            r = auc_scores.get(f'roc_auc_{cls_name}', float('nan'))
            p = auc_scores.get(f'pr_auc_{cls_name}',  float('nan'))
            print(f"  {cls_name:<{col}} {r:>10.4f}   {p:>12.4f}")
        print(f"  {'─'*col} {'─'*10}   {'─'*12}")
        for avg in ('macro', 'micro', 'weighted'):
            label = f"{avg.capitalize()} average"
            print(f"  {label:<{col}} {auc_scores[f'roc_auc_{avg}']:>10.4f}   "
                  f"{auc_scores[f'pr_auc_{avg}']:>12.4f}")
        print(f"{'='*60}")

        results['auc_scores'] = auc_scores
        return auc_scores

    # ------------------------------------------------------------------
    # Confusion matrix
    # ------------------------------------------------------------------

    def _confusion_matrix_figure(
        self,
        cm: np.ndarray,
        class_names: List[str],
        title: str,
        normalized: bool,
    ):
        """Build one confusion-matrix heatmap figure."""
        n = len(class_names)
        data = cm.astype(float)
        if normalized:
            row_sums = data.sum(axis=1, keepdims=True)
            with np.errstate(invalid='ignore', divide='ignore'):
                data = np.divide(data, row_sums, out=np.zeros_like(data), where=row_sums > 0)

        size = max(6.5, 1.15 * n + 2.5)
        fig, ax = plt.subplots(figsize=(size, size * 0.88))
        image = ax.imshow(data, cmap='Blues', vmin=0,
                          vmax=1.0 if normalized else data.max())

        colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        colorbar.set_label('Fraction of true class' if normalized else 'Samples',
                           fontsize=10)

        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(class_names, rotation=45, ha='right', fontsize=10)
        ax.set_yticklabels(class_names, fontsize=10)
        ax.set_xlabel('Predicted label', fontsize=12)
        ax.set_ylabel('True label', fontsize=12)
        ax.set_title(title, fontsize=13, fontweight='bold')

        # Annotate cells; flip text colour on dark cells for readability
        threshold = (1.0 if normalized else data.max()) * 0.55
        fontsize = 10 if n <= 8 else 7
        for i in range(n):
            for j in range(n):
                if normalized:
                    text = f"{data[i, j]*100:.1f}%"
                else:
                    text = f"{int(cm[i, j]):,}"
                ax.text(j, i, text, ha='center', va='center', fontsize=fontsize,
                        color='white' if data[i, j] > threshold else 'black')

        ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
        ax.grid(which='minor', color='white', linewidth=1.2)
        ax.tick_params(which='minor', length=0)
        fig.tight_layout()
        return fig

    def plot_confusion_matrix(
        self,
        results: Dict,
        save_dir: Optional[str] = None,
        show: bool = False,
        dpi: int = 150,
        plot_format: str = 'pdf',
    ) -> Dict[str, List[str]]:
        """
        Plot the confusion matrix twice: raw counts and row-normalized
        (row-normalized diagonal = per-class recall).

        Args:
            results:     Dict returned by evaluate_dataset().
            save_dir:    Output directory.
            show:        Display the figures.
            dpi:         PNG resolution.
            plot_format: 'pdf', 'png' or 'both'.

        Returns:
            Dict 'counts'/'normalized' -> written paths.
        """
        cm = results.get('_confusion_matrix')
        if cm is None:
            cm = np.array(results['confusion_matrix'])
        split = results.get('split', 'split')
        class_names = results.get('class_names') or self._class_names(split)
        accuracy = results.get('accuracy')

        print(f"\nConfusion matrix plots [split={split.upper()}]")

        suffix = f"  |  accuracy = {accuracy:.4f}" if accuracy is not None else ""
        fig = self._confusion_matrix_figure(
            cm, class_names,
            f"Confusion Matrix — {split.upper()}{suffix}",
            normalized=False,
        )
        written = {'counts': save_figure(fig, save_dir, f"confusion_matrix_{split}",
                                         plot_format, dpi, show)}

        balanced = results.get('balanced_accuracy')
        suffix = f"  |  balanced accuracy = {balanced:.4f}" if balanced is not None else ""
        fig = self._confusion_matrix_figure(
            cm, class_names,
            f"Confusion Matrix (row-normalized) — {split.upper()}{suffix}",
            normalized=True,
        )
        written['normalized'] = save_figure(
            fig, save_dir, f"confusion_matrix_normalized_{split}",
            plot_format, dpi, show)

        return written

    # ------------------------------------------------------------------
    # Confidence histogram
    # ------------------------------------------------------------------

    def plot_confidence_histogram(
        self,
        results: Dict,
        save_dir: Optional[str] = None,
        show: bool = False,
        dpi: int = 150,
        plot_format: str = 'pdf',
        bins: int = 40,
    ) -> List[str]:
        """
        Overlaid confidence histograms for correct vs incorrect predictions.

        A well-behaved model puts correct predictions at high confidence and
        errors at low confidence; overlap in the high-confidence region means
        confidently wrong predictions.
        """
        confidences = results['_confidences']
        correct = results['_predictions'] == results['_targets']
        split = results.get('split', 'split')

        fig, (ax_hist, ax_cum) = plt.subplots(1, 2, figsize=(13, 5))

        edges = np.linspace(confidences.min(), 1.0, bins + 1)
        ax_hist.hist(confidences[correct], bins=edges, color='#2ca02c', alpha=0.65,
                     label=f"Correct (n={int(correct.sum()):,})")
        ax_hist.hist(confidences[~correct], bins=edges, color='#d62728', alpha=0.65,
                     label=f"Incorrect (n={int((~correct).sum()):,})")
        if correct.any():
            ax_hist.axvline(confidences[correct].mean(), color='#2ca02c',
                            linestyle='--', lw=1.5,
                            label=f"mean correct = {confidences[correct].mean():.4f}")
        if (~correct).any():
            ax_hist.axvline(confidences[~correct].mean(), color='#d62728',
                            linestyle='--', lw=1.5,
                            label=f"mean incorrect = {confidences[~correct].mean():.4f}")
        ax_hist.set_xlabel('Predicted confidence', fontsize=12)
        ax_hist.set_ylabel('Number of samples', fontsize=12)
        ax_hist.set_title('Confidence distribution', fontsize=12, fontweight='bold')
        ax_hist.legend(fontsize=9)
        ax_hist.grid(True, alpha=0.3)

        # Accuracy vs confidence threshold — how much can be auto-accepted
        thresholds = np.linspace(confidences.min(), confidences.max(), 100)
        coverage, selective_acc = [], []
        for threshold in thresholds:
            mask = confidences >= threshold
            coverage.append(mask.mean())
            selective_acc.append(correct[mask].mean() if mask.any() else np.nan)
        ax_cum.plot(thresholds, selective_acc, color='#1f77b4', lw=2,
                    label='Accuracy above threshold')
        ax_cum.plot(thresholds, coverage, color='#ff7f0e', lw=2, linestyle='--',
                    label='Coverage (fraction kept)')
        ax_cum.set_xlabel('Confidence threshold', fontsize=12)
        ax_cum.set_ylabel('Accuracy / coverage', fontsize=12)
        ax_cum.set_title('Selective prediction', fontsize=12, fontweight='bold')
        ax_cum.legend(fontsize=9, loc='center left')
        ax_cum.grid(True, alpha=0.3)

        fig.suptitle(f"Confidence analysis — {split.upper()}", fontsize=13, fontweight='bold')
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        print(f"\nConfidence histogram [split={split.upper()}]")
        return save_figure(fig, save_dir, f"confidence_hist_{split}",
                           plot_format, dpi, show)

    # ------------------------------------------------------------------
    # Per-class metrics / class distribution
    # ------------------------------------------------------------------

    def plot_per_class_metrics(
        self,
        results: Dict,
        save_dir: Optional[str] = None,
        show: bool = False,
        dpi: int = 150,
        plot_format: str = 'pdf',
    ) -> List[str]:
        """Grouped bar chart of per-class precision, recall and F1."""
        split = results.get('split', 'split')
        class_names = results.get('class_names') or self._class_names(split)
        per_class = results['per_class']

        names = [n for n in class_names if n in per_class]
        x = np.arange(len(names))
        width = 0.26

        fig, ax = plt.subplots(figsize=(max(8, 1.5 * len(names) + 3), 6))
        series = [
            ('Precision', [per_class[n]['precision'] for n in names], '#1f77b4'),
            ('Recall',    [per_class[n]['recall']    for n in names], '#ff7f0e'),
            ('F1',        [per_class[n]['f1']        for n in names], '#2ca02c'),
        ]
        for offset, (label, values, color) in zip((-width, 0.0, width), series):
            bars = ax.bar(x + offset, values, width, label=label, color=color)
            ax.bar_label(bars, fmt='%.3f', fontsize=7, rotation=90, padding=2)

        macro_f1 = results.get('f1_macro')
        if macro_f1 is not None:
            ax.axhline(macro_f1, color='black', linestyle=':', lw=1.2,
                       label=f"F1 macro = {macro_f1:.4f}")

        ax.set_xticks(x)
        ax.set_xticklabels(
            [f"{n}\n(n={per_class[n]['support']:,})" for n in names],
            fontsize=9
        )
        ax.set_ylim(0, 1.12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title(f"Per-class metrics — {split.upper()}", fontsize=13, fontweight='bold')
        # Below the axes: the bars are tall for every class, so any in-plot
        # legend would cover data.
        ax.legend(fontsize=9, ncol=4, loc='upper center',
                  bbox_to_anchor=(0.5, -0.13), frameon=False)
        ax.grid(True, alpha=0.3, axis='y')
        fig.tight_layout()

        print(f"\nPer-class metrics plot [split={split.upper()}]")
        return save_figure(fig, save_dir, f"per_class_metrics_{split}",
                           plot_format, dpi, show)

    def plot_class_distribution(
        self,
        results: Dict,
        save_dir: Optional[str] = None,
        show: bool = False,
        dpi: int = 150,
        plot_format: str = 'pdf',
    ) -> List[str]:
        """
        True vs predicted counts per class — makes over/under-prediction and the
        dataset imbalance visible at a glance.
        """
        split = results.get('split', 'split')
        class_names = results.get('class_names') or self._class_names(split)
        per_class = results['per_class']
        names = [n for n in class_names if n in per_class]

        support = [per_class[n]['support'] for n in names]
        predicted = [per_class[n]['predicted_count'] for n in names]

        x = np.arange(len(names))
        width = 0.38

        fig, ax = plt.subplots(figsize=(max(8, 1.5 * len(names) + 3), 5.5))
        bars_true = ax.bar(x - width / 2, support, width, label='True (support)',
                           color='#4c72b0')
        bars_pred = ax.bar(x + width / 2, predicted, width, label='Predicted',
                           color='#dd8452')
        ax.bar_label(bars_true, fmt='%d', fontsize=8, padding=2)
        ax.bar_label(bars_pred, fmt='%d', fontsize=8, padding=2)

        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=20, ha='right', fontsize=10)
        ax.set_ylabel('Number of samples', fontsize=12)
        ax.set_title(f"Class distribution: true vs predicted — {split.upper()}",
                     fontsize=13, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        fig.tight_layout()

        print(f"\nClass distribution plot [split={split.upper()}]")
        return save_figure(fig, save_dir, f"class_distribution_{split}",
                           plot_format, dpi, show)

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    def plot_calibration(
        self,
        results: Dict,
        save_dir: Optional[str] = None,
        show: bool = False,
        dpi: int = 150,
        plot_format: str = 'pdf',
    ) -> List[str]:
        """
        Reliability diagram: accuracy vs confidence per bin, with the gap to
        perfect calibration highlighted, plus ECE / MCE in the title.
        """
        split = results.get('split', 'split')
        calibration = results.get('_calibration')
        if calibration is None:
            calibration = self._compute_calibration(
                results['_confidences'],
                results['_predictions'] == results['_targets'],
            )

        edges = calibration['bin_edges']
        centers = (edges[:-1] + edges[1:]) / 2
        widths = np.diff(edges)
        accuracy = calibration['bin_accuracy']
        confidence = calibration['bin_confidence']
        counts = calibration['bin_count']

        fig, (ax_rel, ax_count) = plt.subplots(
            2, 1, figsize=(8, 8), sharex=True,
            gridspec_kw={'height_ratios': [3, 1]}
        )

        ax_rel.plot([0, 1], [0, 1], 'k--', lw=1.2, label='Perfect calibration')
        valid = ~np.isnan(accuracy)
        ax_rel.bar(centers[valid], accuracy[valid], width=widths[valid] * 0.92,
                   color='#4c72b0', edgecolor='white', label='Accuracy', zorder=3)
        # The gap can point either way (over- or underconfident), so draw it
        # unfilled: a translucent fill over the solid accuracy bar would be
        # ambiguous exactly in the underconfident case.
        gap = np.where(valid, np.nan_to_num(confidence) - np.nan_to_num(accuracy), np.nan)
        ax_rel.bar(centers[valid], gap[valid], width=widths[valid] * 0.92,
                   bottom=accuracy[valid], fill=False, edgecolor='#d62728',
                   linewidth=0.9, hatch='//', label='Gap (conf − acc)', zorder=5)
        ax_rel.plot(confidence[valid], accuracy[valid], 'o-', color='#2ca02c',
                    lw=1.4, ms=4, label='Bin (conf, acc)', zorder=6)

        ax_rel.set_xlim(0, 1)
        ax_rel.set_ylim(0, 1.02)
        ax_rel.set_ylabel('Accuracy', fontsize=12)
        # Signed gap tells over- from underconfidence, which the unsigned ECE cannot.
        signed_gap = calibration['signed_gap']
        tendency = 'overconfident' if signed_gap > 0 else 'underconfident'
        ax_rel.set_title(
            f"Reliability diagram — {split.upper()}  |  "
            f"ECE = {calibration['ece']:.4f}   MCE = {calibration['mce']:.4f}\n"
            f"mean signed gap = {signed_gap:+.4f}  ({tendency})",
            fontsize=12, fontweight='bold'
        )
        ax_rel.legend(fontsize=9, loc='upper left')
        ax_rel.grid(True, alpha=0.3)

        ax_count.bar(centers, counts, width=widths * 0.92, color='#7f7f7f')
        ax_count.set_yscale('log')
        ax_count.set_xlabel('Predicted confidence', fontsize=12)
        ax_count.set_ylabel('Samples\n(log)', fontsize=10)
        ax_count.grid(True, alpha=0.3)

        fig.tight_layout()
        print(f"\nCalibration plot [split={split.upper()}]  "
              f"ECE={calibration['ece']:.4f}  MCE={calibration['mce']:.4f}")
        return save_figure(fig, save_dir, f"calibration_{split}",
                           plot_format, dpi, show)

    # ------------------------------------------------------------------
    # Misclassified examples
    # ------------------------------------------------------------------

    def plot_misclassified(
        self,
        results: Dict,
        save_dir: Optional[str] = None,
        show: bool = False,
        dpi: int = 150,
        plot_format: str = 'pdf',
        top_n: int = 16,
        order: str = 'confidence',
    ) -> List[str]:
        """
        Grid of the worst misclassifications, i.e. the errors the model was most
        confident about — these are usually the informative ones.

        Args:
            results: Dict returned by evaluate_dataset().
            top_n:   How many images to show (grid is sized automatically).
            order:   'confidence' -> most confident errors first (default);
                     'random'     -> a random sample of errors.

        Note: relies on the evaluation dataloader being built with
        shuffle=False, so sample position == dataset index.
        """
        split = results.get('split', 'split')
        if split not in self.datasets:
            print(f"⚠️  Split '{split}' is no longer loaded — cannot fetch images.")
            return []

        predictions = results['_predictions']
        targets = results['_targets']
        confidences = results['_confidences']
        indices = results['_indices']
        class_names = results.get('class_names') or self._class_names(split)

        wrong = np.flatnonzero(predictions != targets)
        if wrong.size == 0:
            print(f"✓ No misclassified samples on split '{split}' — nothing to plot.")
            return []

        if order == 'random':
            rng = np.random.default_rng(42)
            selected = rng.choice(wrong, size=min(top_n, wrong.size), replace=False)
        else:
            selected = wrong[np.argsort(-confidences[wrong])][:top_n]

        n = len(selected)
        n_cols = int(np.ceil(np.sqrt(n)))
        n_rows = int(np.ceil(n / n_cols))

        dataset = self.datasets[split]
        fig, axes = plt.subplots(n_rows, n_cols,
                                 figsize=(3.0 * n_cols, 3.4 * n_rows))
        axes = np.atleast_1d(axes).ravel()

        for ax, position in zip(axes, selected):
            dataset_index = int(indices[position])
            try:
                image_tensor, _ = dataset[dataset_index]
                ax.imshow(self.tensor_to_image(image_tensor),
                          cmap=None if image_tensor.shape[0] == 3 else 'gray')
            except Exception as exc:
                ax.text(0.5, 0.5, f"image unavailable\n{exc}", ha='center',
                        va='center', fontsize=7, wrap=True)
            true_name = class_names[int(targets[position])]
            pred_name = class_names[int(predictions[position])]
            ax.set_title(
                f"true: {true_name}\npred: {pred_name} ({confidences[position]:.3f})",
                fontsize=9, color='#b22222'
            )
            ax.axis('off')

        for ax in axes[n:]:
            ax.axis('off')

        error_rate = wrong.size / len(targets)
        fig.suptitle(
            f"Most confident misclassifications — {split.upper()}  |  "
            f"{wrong.size:,} errors of {len(targets):,} ({error_rate*100:.2f}%)",
            fontsize=13, fontweight='bold'
        )
        # h_pad keeps the two-line titles off the images in the row above:
        # with axis('off') tight_layout does not reserve room for them.
        fig.tight_layout(rect=(0, 0, 1, 0.96), h_pad=3.0)

        print(f"\nMisclassified examples plot [split={split.upper()}] — "
              f"showing {n} of {wrong.size:,} errors")
        return save_figure(fig, save_dir, f"misclassified_{split}",
                           plot_format, dpi, show)

    # ------------------------------------------------------------------
    # Plot everything
    # ------------------------------------------------------------------

    def plot_all(
        self,
        results: Dict,
        save_dir: Optional[str] = None,
        show: bool = False,
        dpi: int = 150,
        plot_format: str = 'pdf',
        top_n_misclassified: int = 16,
    ) -> Dict[str, Any]:
        """
        Produce every evaluation plot for one split.  Each plot failure is
        reported but does not abort the rest.
        """
        written: Dict[str, Any] = {}
        tasks = [
            ('curves', lambda: self.plot_curves(results, save_dir, show, dpi, plot_format)),
            ('confusion_matrix', lambda: self.plot_confusion_matrix(
                results, save_dir, show, dpi, plot_format)),
            ('per_class_metrics', lambda: self.plot_per_class_metrics(
                results, save_dir, show, dpi, plot_format)),
            ('class_distribution', lambda: self.plot_class_distribution(
                results, save_dir, show, dpi, plot_format)),
            ('confidence_hist', lambda: self.plot_confidence_histogram(
                results, save_dir, show, dpi, plot_format)),
            ('calibration', lambda: self.plot_calibration(
                results, save_dir, show, dpi, plot_format)),
            ('misclassified', lambda: self.plot_misclassified(
                results, save_dir, show, dpi, plot_format, top_n_misclassified)),
        ]
        for name, task in tasks:
            try:
                written[name] = task()
            except Exception as exc:
                print(f"⚠️  Plot '{name}' failed: {exc}")
        return written

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    @staticmethod
    def _json_safe(value):
        """Recursively convert numpy scalars/arrays into JSON-friendly types."""
        if isinstance(value, dict):
            return {k: Inference._json_safe(v) for k, v in value.items()
                    if not str(k).startswith('_')}
        if isinstance(value, (list, tuple)):
            return [Inference._json_safe(v) for v in value]
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            value = float(value)
        if isinstance(value, (np.bool_,)):
            return bool(value)
        # json.dump would write NaN / Infinity, which is not valid JSON.
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return value

    def _write_json(self, results: Dict, path: str) -> str:
        payload = {k: self._json_safe(v) for k, v in results.items()
                   if not k.startswith('_')}
        directory = os.path.dirname(os.path.abspath(path))
        os.makedirs(directory, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as fh:
            json.dump(payload, fh, indent=2)
        print(f"  ✓ saved → {path}")
        return path

    def export_results(
        self,
        results: Dict,
        out_dir: str,
        save_predictions: bool = False,
    ) -> Dict[str, str]:
        """
        Write the metrics of one split to disk.

        Produces:
            metrics_<split>.json               every scalar + confusion matrix + per-class
            per_class_metrics_<split>.csv      one row per class
            classification_report_<split>.txt  sklearn text report
            predictions_<split>.csv            per-sample predictions (opt-in)

        Returns:
            Dict artefact-name -> path.
        """
        split = results.get('split', 'split')
        class_names = results.get('class_names') or self._class_names(split)
        os.makedirs(out_dir, exist_ok=True)
        written: Dict[str, str] = {}

        print(f"\nExporting results [split={split.upper()}] → {out_dir}")

        written['json'] = self._write_json(
            results, os.path.join(out_dir, f"metrics_{split}.json"))

        # Per-class CSV
        per_class = results['per_class']
        columns = ['precision', 'recall', 'f1', 'specificity', 'accuracy',
                   'support', 'predicted_count', 'true_positives',
                   'false_positives', 'false_negatives', 'mean_confidence',
                   'roc_auc', 'ap']
        path = os.path.join(out_dir, f"per_class_metrics_{split}.csv")
        with open(path, 'w', newline='', encoding='utf-8') as fh:
            writer = csv.writer(fh)
            writer.writerow(['class'] + columns)
            for name in class_names:
                stats = per_class.get(name)
                if stats is None:
                    continue
                writer.writerow([name] + [stats.get(c, '') for c in columns])
        print(f"  ✓ saved → {path}")
        written['per_class_csv'] = path

        # Text classification report
        path = os.path.join(out_dir, f"classification_report_{split}.txt")
        with open(path, 'w', encoding='utf-8') as fh:
            fh.write(f"Split: {split}\n")
            fh.write(f"Model: {results.get('model_path')}\n")
            fh.write(f"Samples: {results['n_samples']}\n\n")
            fh.write(classification_report(
                results['_targets'], results['_predictions'],
                labels=list(range(self.num_classes)),
                target_names=class_names, zero_division=0, digits=4))
            fh.write("\nConfusion matrix (rows = true, cols = predicted):\n")
            fh.write(f"{class_names}\n")
            fh.write(np.array2string(results['_confusion_matrix'], max_line_width=200))
            fh.write("\n")
        print(f"  ✓ saved → {path}")
        written['report_txt'] = path

        # Per-sample predictions
        if save_predictions:
            path = os.path.join(out_dir, f"predictions_{split}.csv")
            probabilities = results['_probabilities']
            has_probs = probabilities is not None and len(probabilities)
            with open(path, 'w', newline='', encoding='utf-8') as fh:
                writer = csv.writer(fh)
                header = ['index', 'target', 'target_name', 'prediction',
                          'prediction_name', 'correct', 'confidence']
                if has_probs:
                    header += [f"prob_{name}" for name in class_names]
                writer.writerow(header)
                for i in range(len(results['_targets'])):
                    target = int(results['_targets'][i])
                    prediction = int(results['_predictions'][i])
                    row = [int(results['_indices'][i]), target, class_names[target],
                           prediction, class_names[prediction],
                           int(target == prediction),
                           f"{results['_confidences'][i]:.6f}"]
                    if has_probs:
                        row += [f"{p:.6f}" for p in probabilities[i]]
                    writer.writerow(row)
            print(f"  ✓ saved → {path}")
            written['predictions_csv'] = path

        return written


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Run inference, evaluation and reporting on the RSCDMamba model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- data / model (not required when only plotting training curves) ---
    parser.add_argument('--model-path', type=str, default=None,
                        help='Path to model checkpoint (required unless only --train-log is used)')
    parser.add_argument('--data-path', type=str, default=None,
                        help='Path to dataset root directory (required unless only --train-log is used)')
    parser.add_argument('--data-set', type=str, default='RSCD',
                        choices=['RSCD', 'IMNET', 'CIFAR', 'INAT', 'INAT19'],
                        help='Dataset type')
    parser.add_argument('--img-size', type=int, default=224,
                        help='Input image size')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for evaluation')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--from-checkpoint', action='store_true',
                        help='Load model from checkpoint format')

    # --- model architecture (must match the trained checkpoint) ---
    parser.add_argument('--num-classes', type=int, default=6,
                        help='Number of output classes of the checkpoint')
    parser.add_argument('--depths', type=int, nargs='+', default=[2, 4, 8, 4],
                        help='Depth of each RSCDMamba stage')
    parser.add_argument('--dims', type=int, default=64,
                        help='Base embedding dimension')
    parser.add_argument('--drop-path-rate', type=float, default=0.3,
                        help='Stochastic depth rate used at training time')
    parser.add_argument('--fg-mask', type=str, default='auto',
                        choices=['auto', 'on', 'off'],
                        help='Foreground mask in the SSM blocks. "auto" enables it only '
                             'if the checkpoint contains the fg_mask weights')
    parser.add_argument('--condition-branch', type=str, default='auto',
                        choices=['auto', 'on', 'off'],
                        help='Auxiliary condition branch. "auto" enables it only if the '
                             'checkpoint contains the condition_branch weights')
    parser.add_argument('--non-strict', action='store_true',
                        help='Allow missing/unexpected checkpoint keys (only warn). '
                             'By default a mismatch aborts, since missing weights '
                             'would silently stay randomly initialized')

    # --- dataset split ---
    parser.add_argument('--split', type=str, default='val',
                        choices=['val', 'test', 'both', 'separate'],
                        help=(
                            'Which dataset split to use for evaluation/inference. '
                            '"val"      -> validation set (is_train=False, is_test=False). '
                            '"test"     -> test set       (is_train=False, is_test=True). '
                            '"both"     -> val + test concatenated into one dataset. '
                            '"separate" -> val AND test evaluated independently and '
                            'reported side by side (like running the script twice).'
                        ))
    parser.add_argument('--class-names', type=str, nargs='+', default=None,
                        help='Override class names (default: ImageFolder folder names)')

    # --- GPU options ---
    parser.add_argument('--gpu-id', type=int, default=None,
                        help='GPU ID to use (default: auto-select)')
    parser.add_argument('--device', type=str, default=None,
                        choices=['cpu', 'cuda'],
                        help='Force device type')

    # --- analysis options ---
    parser.add_argument('--model-summary', action='store_true',
                        help='Model complexity (params / FLOPs / MACs), printed and '
                             'saved as model_summary_mamba.json/.txt')
    parser.add_argument('--benchmark', action='store_true',
                        help='Forward-pass latency benchmark on a synthetic input')
    parser.add_argument('--memory', action='store_true',
                        help='Run memory profiling')
    parser.add_argument('--evaluate', action='store_true',
                        help='Evaluate on the selected dataset split(s)')
    parser.add_argument('--inference', action='store_true',
                        help='Single-image (batch=1) inference latency on real '
                             'images, saved as single_inference_mamba.json/.txt/.csv')
    parser.add_argument('--num-samples', type=int, default=50,
                        help='Number of images for --inference')
    parser.add_argument('--warmup', type=int, default=10,
                        help='Warm-up runs excluded from the --inference timings')
    parser.add_argument('--seed', type=int, default=42,
                        help='Seed for the random image pick of --inference')
    parser.add_argument('--amp', action='store_true',
                        help='Run every forward pass under bf16 autocast (CUDA only)')

    # --- plotting ---
    parser.add_argument('--plot-format', type=str, default='pdf',
                        choices=list(PLOT_FORMATS),
                        help='Output format for every figure')
    parser.add_argument('--plot-dir', type=str, default=None,
                        help='Directory where plots are saved (default: folder of the checkpoint)')
    parser.add_argument('--dpi', type=int, default=150,
                        help='Raster resolution (PNG only)')
    parser.add_argument('--show', action='store_true',
                        help='Display figures interactively (needs a display)')
    parser.add_argument('--plot-all', action='store_true',
                        help='Produce every evaluation plot')
    parser.add_argument('--plot-curves', action='store_true',
                        help='Plot AUC-ROC and AUC-PR curves')
    parser.add_argument('--plot-confusion-matrix', action='store_true',
                        help='Plot the confusion matrix (raw counts + row-normalized)')
    parser.add_argument('--plot-per-class', action='store_true',
                        help='Plot per-class metric bars and class distribution')
    parser.add_argument('--plot-confidence', action='store_true',
                        help='Plot confidence histograms and selective-prediction curves')
    parser.add_argument('--plot-calibration', action='store_true',
                        help='Plot the reliability diagram (ECE / MCE)')
    parser.add_argument('--plot-misclassified', action='store_true',
                        help='Plot a grid of the most confident misclassifications')
    parser.add_argument('--num-misclassified', type=int, default=16,
                        help='How many misclassified images to show')

    # --- training curves ---
    parser.add_argument('--train-log', type=str, nargs='+', default=None,
                        help=('Training log file(s), glob, or a logs/ directory. '
                              'Plots train/validation loss, accuracy, F1, learning '
                              'rate and confidence across epochs. Resumed runs '
                              '(several files) are merged by epoch number.'))
    parser.add_argument('--log-scale-loss', action='store_true',
                        help='Use a logarithmic y-axis for the loss curve')

    # --- export ---
    parser.add_argument('--export-dir', type=str, default=None,
                        help='Directory for JSON/CSV/TXT metric exports (default: --plot-dir)')
    parser.add_argument('--save-predictions', action='store_true',
                        help='Also export per-sample predictions as CSV (can be large)')

    return parser


def resolve_device(args) -> torch.device:
    """Pick the device from the CLI flags."""
    if args.device == 'cpu':
        print("⚠️ Forcing CPU usage")
        return torch.device('cpu')

    if args.gpu_id is not None:
        if not torch.cuda.is_available():
            print("⚠️ CUDA not available, falling back to CPU")
            return torch.device('cpu')
        if args.gpu_id >= torch.cuda.device_count():
            raise ValueError(
                f"GPU {args.gpu_id} not available. "
                f"Available GPUs: 0-{torch.cuda.device_count()-1}"
            )
        print(f"✓ Using GPU {args.gpu_id}: {torch.cuda.get_device_name(args.gpu_id)}")
        return torch.device(f'cuda:{args.gpu_id}')

    return Inference.get_best_device(
        force_cpu=(args.device == "cpu"),
        gpu_id=args.gpu_id
    )


def main():
    """Command-line entry point for the Inference class."""
    parser = build_parser()
    args = parser.parse_args()

    wants_model = any([
        args.evaluate, args.inference, args.model_summary,
        args.benchmark, args.memory,
    ])
    # Evaluation-only plot flags imply --evaluate
    wants_eval_plots = any([
        args.plot_all, args.plot_curves, args.plot_confusion_matrix,
        args.plot_per_class, args.plot_confidence, args.plot_calibration,
        args.plot_misclassified,
    ])

    if not wants_model and not wants_eval_plots and not args.train_log:
        parser.error(
            "Nothing to do. Pick at least one action: --evaluate, --inference, "
            "--model-summary, --benchmark, --memory, a --plot-* flag, or --train-log."
        )

    if wants_eval_plots and not args.evaluate:
        print("ℹ️  Plot flags require an evaluation — enabling --evaluate")
        args.evaluate = True
        wants_model = True

    if wants_model:
        missing = [name for name, value in (('--model-path', args.model_path),
                                            ('--data-path', args.data_path))
                   if not value]
        if missing:
            parser.error(
                f"{', '.join(missing)} required for the requested action(s). "
                "Use only --train-log if you just want the training curves."
            )

    plot_dir = args.plot_dir
    if plot_dir is None and args.model_path:
        plot_dir = os.path.dirname(os.path.abspath(args.model_path))
    export_dir = args.export_dir or plot_dir

    # ── Training curves from the log (no model or dataset needed) ────
    history = None
    if args.train_log:
        try:
            history = TrainingLogParser.parse(args.train_log, verbose=True)
            plot_training_curves(
                history,
                save_dir=plot_dir,
                plot_format=args.plot_format,
                show=args.show,
                dpi=args.dpi,
                log_scale=args.log_scale_loss,
            )
            if args.export_dir:
                history.to_csv(os.path.join(args.export_dir, 'training_history.csv'))
        except (FileNotFoundError, ValueError) as exc:
            print(f"⚠️  Training curves skipped: {exc}")

    if not wants_model:
        print("\n✓ Done.")
        return

    device = resolve_device(args)

    def _resolve_flag(mode: str, state_dict, marker: str) -> bool:
        if mode == 'on':
            return True
        if mode == 'off':
            return False
        if state_dict is None:
            return True  # model default
        return any(marker in key for key in state_dict.keys())

    def build_model(state_dict=None) -> torch.nn.Module:
        """Build RSCDMamba, inferring optional modules from the checkpoint keys."""
        use_fg_mask = _resolve_flag(args.fg_mask, state_dict, '.fg_mask.')
        use_condition_branch = _resolve_flag(args.condition_branch, state_dict, 'condition_branch.')
        print(f"\nCreating model instance (use_fg_mask={use_fg_mask}, "
              f"use_condition_branch={use_condition_branch})...")
        return RSCDMamba(
            img_size=args.img_size,
            num_classes=args.num_classes,
            depths=args.depths,
            dims=args.dims,
            drop_path_rate=args.drop_path_rate,
            use_fg_mask=use_fg_mask,
            use_condition_branch=use_condition_branch,
        )

    # With --from-checkpoint the model is built once the state dict is known
    model_instance = build_model if args.from_checkpoint else None

    print(f"Initializing inference (split={args.split})...")
    inference = Inference(
        model_path=args.model_path,
        data_path=args.data_path,
        data_set=args.data_set,
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        load_from_checkpoint=args.from_checkpoint,
        model_instance=model_instance,
        device=device,
        split=args.split,
        class_names=args.class_names,
        amp=args.amp,
        strict=not args.non_strict,
    )

    if args.model_summary or args.benchmark or args.memory:
        summary = inference.model_summary(
            include_benchmark=args.benchmark,
            include_memory=args.memory
        )
        if export_dir:
            inference.export_model_summary(summary, export_dir)

    if args.evaluate:
        results_by_split = inference.evaluate_splits(
            verbose=True,
            save_predictions=args.save_predictions,
        )
        print("\n✓ Evaluation complete!")

        for split_name, results in results_by_split.items():
            if args.plot_all:
                inference.plot_all(
                    results, save_dir=plot_dir, show=args.show, dpi=args.dpi,
                    plot_format=args.plot_format,
                    top_n_misclassified=args.num_misclassified,
                )
            else:
                if args.plot_curves:
                    inference.plot_curves(results, plot_dir, args.show,
                                          args.dpi, args.plot_format)
                if args.plot_confusion_matrix:
                    inference.plot_confusion_matrix(results, plot_dir, args.show,
                                                    args.dpi, args.plot_format)
                if args.plot_per_class:
                    inference.plot_per_class_metrics(results, plot_dir, args.show,
                                                     args.dpi, args.plot_format)
                    inference.plot_class_distribution(results, plot_dir, args.show,
                                                      args.dpi, args.plot_format)
                if args.plot_confidence:
                    inference.plot_confidence_histogram(results, plot_dir, args.show,
                                                        args.dpi, args.plot_format)
                if args.plot_calibration:
                    inference.plot_calibration(results, plot_dir, args.show,
                                               args.dpi, args.plot_format)
                if args.plot_misclassified:
                    inference.plot_misclassified(
                        results, plot_dir, args.show, args.dpi,
                        args.plot_format, args.num_misclassified)

            if export_dir:
                inference.export_results(results, export_dir,
                                         save_predictions=args.save_predictions)

        if len(results_by_split) > 1 and export_dir:
            inference.print_split_comparison(results_by_split, out_dir=export_dir)

    if args.inference:
        summary = inference.single_image_inference(
            num_samples=args.num_samples,
            warmup_runs=args.warmup,
            seed=args.seed,
        )
        if export_dir:
            inference.export_single_inference(summary, export_dir)

    print("\n✓ Done.")


if __name__ == "__main__":
    main()
