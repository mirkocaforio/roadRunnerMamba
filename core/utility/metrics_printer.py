"""
MetricsPrinter — pretty-print the metrics dict returned by
CrossEntropyLossStrategy.compute_metrics() and SigmoidFocalLossStrategy.compute_metrics().

Drop this class into loss.py (or import it from a separate module) and call:

    printer = MetricsPrinter(class_names=['dry', 'fresh_snow', ...])
    printer.print(metrics, prefix='val')
"""

import numpy as np


class MetricsPrinter:
    """
    Formats and prints the metrics dictionary produced by
    CrossEntropyLossStrategy and SigmoidFocalLossStrategy.

    Args:
        class_names:  List of human-readable class names (optional).
                      If omitted, class_0 / class_1 … labels are used.
        width:        Total width of the printed block (default 68).
        pct_decimals: Decimal places for percentage values (default 2).
        float_decimals: Decimal places for raw float values (default 4).
    """

    # Thresholds for coloring accuracy / F1 values
    _GOOD  = 0.90
    _WARN  = 0.75

    # ANSI color codes (disabled automatically when NO_COLOR env-var is set)
    _GREEN  = "\033[92m"
    _YELLOW = "\033[93m"
    _RED    = "\033[91m"
    _CYAN   = "\033[96m"
    _BOLD   = "\033[1m"
    _DIM    = "\033[2m"
    _RESET  = "\033[0m"

    def __init__(
        self,
        class_names=None,
        width: int = 68,
        pct_decimals: int = 2,
        float_decimals: int = 4,
        use_color: bool = True,
    ):
        self.class_names   = class_names
        self.width         = width
        self.pct           = pct_decimals
        self.flt           = float_decimals
        self.use_color     = use_color

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def print(self, metrics: dict, prefix: str = "val", loss: float = None):
        """
        Print the full metrics report.

        Args:
            metrics: Dict returned by compute_metrics(detailed=True).
            prefix:  String prepended to section headers ("val" or "train").
            loss:    Optional loss value to show in the header.
        """
        lines = self._build(metrics, prefix, loss)
        print("\n".join(lines))

    def format(self, metrics: dict, prefix: str = "val", loss: float = None) -> str:
        """Same as print() but returns a string instead."""
        return "\n".join(self._build(metrics, prefix, loss))

    # ------------------------------------------------------------------
    # Internal builder
    # ------------------------------------------------------------------

    def _build(self, metrics: dict, prefix: str, loss: float) -> list:
        lines = []
        W = self.width

        lines += self._header(f" {prefix.upper()} METRICS ", W)

        # ── Core classification metrics ────────────────────────────────
        lines.append(self._section("core metrics"))
        core = [
            ("accuracy",          metrics.get("accuracy")),
            ("balanced accuracy", metrics.get("balanced_accuracy")),
            ("F1  macro",         metrics.get("f1_macro")),
            ("F1  weighted",      metrics.get("f1_weighted")),
            ("F1  micro",         metrics.get("f1_micro")),
            ("precision  macro",  metrics.get("precision_macro")),
            ("precision  weighted", metrics.get("precision_weighted")),
            ("recall  macro",     metrics.get("recall_macro")),
            ("recall  weighted",  metrics.get("recall_weighted")),
        ]
        if loss is not None:
            lines.append(self._kv("loss", f"{loss:.6f}", W))
        for label, val in core:
            if val is not None:
                lines.append(self._kv_pct(label, val, W))

        # ── Agreement metrics ──────────────────────────────────────────
        lines.append(self._section("agreement"))
        for label, key in [("cohen κ", "cohen_kappa"), ("matthews MCC", "matthews_corrcoef")]:
            val = metrics.get(key)
            if val is not None:
                lines.append(self._kv(label, f"{val:.{self.flt}f}", W))

        # ── Top-K accuracy ─────────────────────────────────────────────
        topk_items = [(k, v) for k, v in metrics.items() if k.startswith("top_") and k.endswith("_accuracy")]
        if topk_items:
            lines.append(self._section("top-k accuracy"))
            for key, val in sorted(topk_items):
                k = key.split("_")[1]
                lines.append(self._kv_pct(f"top-{k}", val, W))

        # ── Confidence ─────────────────────────────────────────────────
        conf_keys = [
            ("mean confidence",            "mean_confidence"),
            ("median confidence",          "median_confidence"),
            ("min confidence",             "min_confidence"),
            ("mean confidence (correct)",  "mean_correct_confidence"),
            ("mean confidence (incorrect)","mean_incorrect_confidence"),
        ]
        conf_items = [(lbl, metrics[key]) for lbl, key in conf_keys if key in metrics]
        if conf_items:
            lines.append(self._section("confidence"))
            for label, val in conf_items:
                lines.append(self._kv_pct(label, val, W))

        # ── Per-class breakdown ────────────────────────────────────────
        per_class = metrics.get("per_class")
        if per_class:
            lines.append(self._section("per-class breakdown"))
            lines += self._per_class_table(per_class, W)

        # ── Confusion matrix ───────────────────────────────────────────
        cm = metrics.get("confusion_matrix")
        if cm is not None:
            lines.append(self._section("confusion matrix"))
            lines += self._confusion_matrix(cm, per_class, W)

        lines += self._footer(W)
        return lines

    # ------------------------------------------------------------------
    # Formatting helpers
    # ------------------------------------------------------------------

    def _c(self, text: str, code: str) -> str:
        if not self.use_color:
            return text
        return f"{code}{text}{self._RESET}"

    def _color_for(self, value: float) -> str:
        if value >= self._GOOD:
            return self._GREEN
        if value >= self._WARN:
            return self._YELLOW
        return self._RED

    def _pct_str(self, value: float) -> str:
        return f"{value * 100:.{self.pct}f}%"

    def _header(self, title: str, W: int) -> list:
        bar = "─" * W
        mid = title.center(W)
        return [
            self._c(f"╔{'═' * W}╗", self._BOLD),
            self._c(f"║{mid}║", self._BOLD),
            self._c(f"╚{'═' * W}╝", self._BOLD),
        ]

    def _footer(self, W: int) -> list:
        return [self._c("─" * W, self._DIM)]

    def _section(self, title: str) -> str:
        label = f"  {title.upper()}  "
        return self._c(f"\n  ┌─{label}{'─' * max(0, self.width - len(label) - 4)}┐", self._DIM)

    def _kv(self, key: str, val_str: str, W: int) -> str:
        """Left-aligned key, right-aligned value, fixed total width."""
        gap = W - len(key) - len(val_str) - 4
        return f"  │ {self._c(key, self._DIM)}{' ' * max(1, gap)}{val_str} │"

    def _kv_pct(self, key: str, val: float, W: int) -> str:
        val_str = self._c(self._pct_str(val), self._color_for(val))
        # Strip ANSI when computing padding
        raw_val_str = self._pct_str(val)
        gap = W - len(key) - len(raw_val_str) - 4
        return f"  │ {self._c(key, self._DIM)}{' ' * max(1, gap)}{val_str} │"

    # ------------------------------------------------------------------
    # Per-class table
    # ------------------------------------------------------------------

    def _per_class_table(self, per_class: dict, W: int) -> list:
        # Column widths
        name_w = max(len(n) for n in per_class) + 2
        col_w  = 9   # "100.00%"
        cols   = ["precision", "recall", "F1", "support", "confidence"]

        header_parts = [f"{'class':<{name_w}}"]
        for c in cols:
            header_parts.append(f"{c:>{col_w}}")
        header = "  │  " + "  ".join(header_parts) + "  │"

        sep = "  │  " + "─" * (len(header) - 10) + "  │"
        lines = [self._c(header, self._DIM), sep]

        for class_key, data in per_class.items():
            # Prefer explicit class_names list; fall back to key
            name = class_key  # keys are already class names in compute_metrics
            prec  = data.get("precision", 0.0)
            rec   = data.get("recall",    0.0)
            f1    = data.get("f1",        0.0)
            sup   = data.get("support",   0)
            conf  = data.get("mean_confidence")

            row_parts = [f"{name:<{name_w}}"]
            for val in (prec, rec, f1):
                s = self._pct_str(val)
                row_parts.append(f"{self._c(s, self._color_for(val)):>{col_w + len(self._c('', self._color_for(val)))}}")
            row_parts.append(f"{sup:>{col_w},}")
            if conf is not None:
                s = self._pct_str(conf)
                row_parts.append(f"{self._c(s, self._color_for(conf)):>{col_w + len(self._c('', self._color_for(conf)))}}")
            else:
                row_parts.append(f"{'N/A':>{col_w}}")

            lines.append("  │  " + "  ".join(row_parts) + "  │")

        return lines

    # ------------------------------------------------------------------
    # Confusion matrix
    # ------------------------------------------------------------------

    def _confusion_matrix(self, cm, per_class, W: int) -> list:
        cm = np.array(cm)
        n  = cm.shape[0]

        # Class labels: use per_class keys if available, else generic
        if per_class:
            labels = list(per_class.keys())
        elif self.class_names:
            labels = self.class_names[:n]
        else:
            labels = [f"c{i}" for i in range(n)]

        # Abbreviate to at most 6 chars for column headers
        short = [lb[:6] for lb in labels]
        col_w = max(6, max(len(s) for s in short)) + 1
        row_label_w = max(len(lb) for lb in labels) + 1

        # Header row
        header = " " * (row_label_w + 2) + "  ".join(f"{s:>{col_w}}" for s in short)
        lines  = [f"  │  {self._c(header, self._DIM)}"]

        sep = "  │  " + "─" * len(header)
        lines.append(sep)

        row_totals = cm.sum(axis=1)
        for i, (lb, row) in enumerate(zip(labels, cm)):
            parts = []
            for j, val in enumerate(row):
                s = str(val)
                if i == j:
                    # Diagonal: correct predictions
                    s = self._c(f"{s:>{col_w}}", self._GREEN)
                elif val == 0:
                    s = self._c(f"{s:>{col_w}}", self._DIM)
                else:
                    # Off-diagonal errors: color by magnitude
                    ratio = val / max(row_totals[i], 1)
                    color = self._RED if ratio > 0.05 else self._YELLOW
                    s = self._c(f"{s:>{col_w}}", color)
                parts.append(s)

            row_str = "  ".join(parts)
            recall_str = self._c(
                f"  {self._pct_str(cm[i, i] / max(row_totals[i], 1))}",
                self._color_for(cm[i, i] / max(row_totals[i], 1))
            )
            lines.append(f"  │  {lb:<{row_label_w}} {row_str}{recall_str}")

        return lines


# ------------------------------------------------------------------
# Convenience: patch compute_metrics to accept a printer kwarg
# ------------------------------------------------------------------

def attach_printer(strategy_instance, class_names=None, **printer_kwargs):
    """
    Monkey-patch a loss strategy so that calling
      strategy.print_metrics(metrics, prefix='val')
    produces pretty output.

    Usage:
        strategy = CrossEntropyLossStrategy(num_classes=6)
        attach_printer(strategy, class_names=['dry', 'fresh_snow', ...])
        ...
        metrics = strategy.compute_metrics(outputs, targets, detailed=True)
        strategy.print_metrics(metrics, prefix='val')
    """
    printer = MetricsPrinter(class_names=class_names, **printer_kwargs)
    strategy.print_metrics = lambda metrics, prefix="val", loss=None: printer.print(metrics, prefix, loss)
    strategy._metrics_printer = printer
    return strategy