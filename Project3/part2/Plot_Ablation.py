import re
import ast
import matplotlib.pyplot as plt
from typing import Union, Dict, Any, List




def plot_metrics_from_file(file_path: str, save_path: str = None) -> plt.Figure:
    """
    Read a text file containing lines of the form:
      3 clicks: {'dice': 0.8317, 'iou': 0.7470, 'acc': 0.8398, 'sen': 0.9304, 'spe': 0.7842}
      5 points: {'dice': ...}
      All: {'dice': ...}
    Parse the values and plot each metric (dice, iou, acc, sen, spe) vs number of clicks.
    'All' (if present) is shown at the far right and treated as a non-numeric category.
    
    Args:
        file_path: path to the text file with one experiment result per line.
        save_path: optional path to save the figure (e.g. "metrics.png"). If None, the figure
                   is just returned and displayed by the caller.
    Returns:
        matplotlib.figure.Figure
    """
    results: Dict[Union[int, str], Dict[str, float]] = {}

    with open(file_path, "r") as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # detect the 'All' case
        if re.match(r"^\s*All\s*:", line, flags=re.IGNORECASE):
            clicks_key = "All"
        else:
            # match an integer followed by 'clicks' or 'points' (case-insensitive)
            m = re.search(r"(\d+)\s*(?:clicks|points)?\s*:", line, flags=re.IGNORECASE)
            if not m:
                # if no numeric clicks and not All, skip line
                continue
            clicks_key = int(m.group(1))

        # extract the dict-like substring {...}
        dict_match = re.search(r"\{.*\}", line)
        if not dict_match:
            continue

        dict_text = dict_match.group(0)

        # parse the dict text safely using ast.literal_eval
        try:
            metrics = ast.literal_eval(dict_text)
        except Exception:
            # fallback: try replacing single quotes with double quotes and re-evaluate
            try:
                metrics = ast.literal_eval(dict_text.replace("'", "\""))
            except Exception:
                # if parsing fails, skip
                continue

        # ensure floats
        parsed_metrics = {}
        for k, v in metrics.items():
            try:
                parsed_metrics[str(k)] = float(v)
            except Exception:
                # skip non-numeric entries
                pass

        if parsed_metrics:
            results[clicks_key] = parsed_metrics

    if not results:
        raise RuntimeError("No valid metric lines parsed from file.")

    # determine metric names (use first entry)
    first_metrics = next(iter(results.values()))
    metric_names = list(first_metrics.keys())

    # sort numeric keys; move 'All' to end if present
    numeric_keys: List[int] = sorted([k for k in results.keys() if isinstance(k, int)])
    keys_order: List[Union[int, str]] = numeric_keys
    if "All" in results:
        keys_order = numeric_keys + ["All"]

    # prepare x ticks and numeric x positions
    x_labels = [str(k) for k in keys_order]
    x_positions = list(range(len(keys_order)))

    # create plot
    fig, ax = plt.subplots(figsize=(9, 5))
    for metric in metric_names:
        y = [results[k][metric] for k in keys_order]
        ax.plot(x_positions, y, marker="o", label=metric)

    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels)
    ax.set_xlabel("Number of clicks")
    ax.set_ylabel("Metric value")
    ax.set_ylim(0.0, 1.05)  # metrics are in [0,1], give a little headroom
    ax.set_title("Segmentation metrics vs. number of clicks")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="best")

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)

    return fig

# -------------------------
# Example usage:
# fig = plot_metrics_from_file("results.txt", save_path="metrics.png")
# plt.show()  # if running in a script or interactive session

plot_metrics_from_file("/zhome/9c/f/221532/Project3/part2/Ablation.txt", save_path="/zhome/9c/f/221532/Project3/part2/Ablation_metrics.png")