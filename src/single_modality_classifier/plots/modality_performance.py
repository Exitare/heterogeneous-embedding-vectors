from pathlib import Path
import pandas as pd
import argparse
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from statannotations.Annotator import Annotator

# ------- Config -------
FIG_ROOT = Path("figures", "single_modality_classifier")
CLASS_ROOT = Path("results", "single_modality_classifier", "classification")

DEFAULT_CANCERS = ["BRCA", "LUAD", "STAD", "BLCA", "COAD", "THCA"]
DEFAULT_MODALITIES = ["rna", "annotations", "mutations", "images"]

METRIC_LABELS = {"accuracy": "Accuracy", "f1": "F1"}

# modality -> (pretty name, color)
MODALITY_LABELS = {
    "rna": ("RNA", "#c6afe9ff"),
    "annotations": ("Annotation", "#c8b7b7ff"),
    "mutations": ("Mutation", "#de87aaff"),
    "images": ("Image", "#d38d5fff"),
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# ------- Data loading -------
def _load_results_for_modality(cancers_key: str, modality: str) -> pd.DataFrame:
    """Load results.csv from each iteration folder for one modality."""
    base = CLASS_ROOT / cancers_key / modality
    if not base.exists():
        logging.warning(f"Skipping modality '{modality}': folder not found: {base}")
        return pd.DataFrame()

    dfs = []
    for run_dir in sorted(base.iterdir()):
        if not run_dir.is_dir():
            continue
        csv_path = run_dir / "results.csv"
        if not csv_path.exists():
            continue
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            logging.warning(f"Failed to read {csv_path}: {e}")
            continue

        try:
            iter_id = int(run_dir.name)
        except ValueError:
            iter_id = run_dir.name
        df["iteration"] = iter_id
        df["modality"] = modality
        dfs.append(df)

    if not dfs:
        logging.warning(f"No results.csv files found under {base}/*/ for modality '{modality}'")
        return pd.DataFrame()

    out = pd.concat(dfs, ignore_index=True)
    out["cancer"] = out["cancer"].astype(str)
    return out


def load_all_modalities(cancers_list: list[str]) -> tuple[pd.DataFrame, str]:
    cancers_key = "_".join(cancers_list)
    frames = [_load_results_for_modality(cancers_key, m) for m in DEFAULT_MODALITIES]
    frames = [f for f in frames if not f.empty]
    if not frames:
        raise FileNotFoundError("No results found for any modality.")
    df = pd.concat(frames, ignore_index=True)

    # Drop "All" rows if they exist
    df = df[df["cancer"] != "All"].copy()

    # Pretty modality labels
    df["modality_pretty"] = df["modality"].map(lambda m: MODALITY_LABELS.get(m, (m, None))[0])

    # Order cancers on x-axis
    desired_order = [c for c in cancers_list if c in df["cancer"].unique()]
    others = [c for c in df["cancer"].unique() if c not in desired_order]
    df["cancer"] = pd.Categorical(df["cancer"], categories=desired_order + others, ordered=True)

    return df, cancers_key


def modality_palette_from_df(df: pd.DataFrame) -> dict:
    palette = {}
    for m in df["modality"].unique():
        pretty, color = MODALITY_LABELS.get(m, (m, None))
        if color:
            palette[pretty] = color
    return palette


# ------- Plotting + annotations -------
def _build_pairs_for_hue_by_x(df: pd.DataFrame, x_col: str, hue_col: str) -> list[tuple[tuple, tuple]]:
    """Build all within-cancer hue pairs."""
    pairs = []
    x_levels = [lvl for lvl in df[x_col].cat.categories if str(lvl) in set(df[x_col].astype(str))]
    for x_val in x_levels:
        sub = df[df[x_col] == x_val]
        hue_levels = sub[hue_col].dropna().unique().tolist()
        hue_levels = sorted(hue_levels, key=lambda h: h)
        for i in range(len(hue_levels)):
            for j in range(i + 1, len(hue_levels)):
                pairs.append(((x_val, hue_levels[i]), (x_val, hue_levels[j])))
    return pairs


def boxenplot_by_modality_with_stats(df: pd.DataFrame, metric: str, save_dir: Path, palette: dict):
    """Boxenplot with Mann–Whitney + BH correction per cancer (no 'All')."""
    if metric not in df.columns:
        logging.warning(f"Metric '{metric}' not found in results; skipping.")
        return

    hue_order = [MODALITY_LABELS[m][0] for m in DEFAULT_MODALITIES
                 if MODALITY_LABELS[m][0] in df["modality_pretty"].unique()]

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxenplot(
        data=df,
        x="cancer",
        y=metric,
        hue="modality_pretty",
        palette=palette if palette else None,
        hue_order=hue_order,
        ax=ax
    )

    ax.set_title(f"{METRIC_LABELS.get(metric, metric)} by Modality (across iterations)", pad=8)
    ax.set_ylabel(METRIC_LABELS.get(metric, metric))
    ax.set_xlabel("Cancer")

    # Compute dynamic headroom
    top99 = float(df[metric].dropna().quantile(0.99)) if df[metric].notna().any() else 1.0
    y_upper = min(1.10, max(0.95, top99) + 0.06)
    ax.set_ylim(0.0, y_upper)

    # Legend below
    ax.legend(title="Modality", ncols=2, loc="upper center",
              bbox_to_anchor=(0.5, -0.12), frameon=False)

    pairs = _build_pairs_for_hue_by_x(df, x_col="cancer", hue_col="modality_pretty")
    annot = Annotator(
        ax=ax,
        pairs=pairs,
        data=df,
        x="cancer",
        y=metric,
        hue="modality_pretty",
        hue_order=hue_order
    )
    annot.configure(
        test="Mann-Whitney",
        comparisons_correction="Benjamini-Hochberg",
        loc="inside",
        text_format="star",
        verbose=1
    )
    annot.apply_and_annotate()

    # remove extra y-ticks > 1.0
    yticks = [yt for yt in ax.get_yticks() if yt <= 1.0]
    ax.set_yticks(yticks)
    ax.set_yticklabels([f"{yt:.2f}" for yt in yticks])

    plt.tight_layout()
    out = save_dir / f"{metric}_by_modality_box_annotated.png"
    plt.savefig(out, dpi=300)
    plt.close(fig)
    logging.info(f"Saved: {out}")


# ------- Main -------
def main():
    parser = argparse.ArgumentParser(
        description="Compare modalities via boxenplots with BH-adjusted Mann–Whitney p-values.")
    parser.add_argument("--cancer", "-c", nargs="+", required=False,
                        default=DEFAULT_CANCERS,
                        help="Cancer types used to build the results path. Defaults to the 6 standard cancers.")
    args = parser.parse_args()

    cancers_list = args.cancer
    logging.info(f"Using cancers: {cancers_list}")

    df, cancers_key = load_all_modalities(cancers_list)
    save_dir = FIG_ROOT / cancers_key / "comparisons"
    save_dir.mkdir(parents=True, exist_ok=True)
    palette = modality_palette_from_df(df)

    boxenplot_by_modality_with_stats(df, "accuracy", save_dir, palette)
    boxenplot_by_modality_with_stats(df, "f1", save_dir, palette)

    logging.info(f"All figures saved to: {save_dir}")


if __name__ == "__main__":
    main()
