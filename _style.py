"""Figure export helper: 600-dpi PNG + audit JSON, no styling mutations."""
from pathlib import Path
import json
import os
from matplotlib.text import Text

ROOT = Path(__file__).resolve().parent


def save_figure(fig, filename, **kwargs):
    stem = Path(filename).stem
    kwargs.setdefault("dpi", 600)
    output = Path(os.environ.get("YREPCMP_FIGURES", str(ROOT / "figures")))
    output.mkdir(parents=True, exist_ok=True)
    target = output / (stem + ".png")
    fig.savefig(target, **kwargs)
    visible = [t for t in fig.findobj(Text) if t.get_visible() and t.get_text().strip()]
    report = {
        "figure": stem,
        "inches": list(fig.get_size_inches()),
        "dpi": kwargs["dpi"],
        "text_sizes_pt": sorted({round(t.get_fontsize(), 2) for t in visible}),
    }
    (output / (stem + ".json")).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[saved] {target}", flush=True)
    return target
