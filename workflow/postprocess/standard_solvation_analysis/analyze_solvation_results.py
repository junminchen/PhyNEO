import argparse
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def load_data(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    required = ["Folder", "Filename", "Category", "Additive", "Charge", "HOMO(eV)", "LUMO(eV)", "Gap(eV)"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    for col in ["Charge", "HOMO(eV)", "LUMO(eV)", "Gap(eV)"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def save_global_summary(df: pd.DataFrame, outdir: Path) -> pd.DataFrame:
    summary = (
        df.groupby("Folder", dropna=False)
        .agg(
            n_structures=("Filename", "count"),
            mean_homo=("HOMO(eV)", "mean"),
            std_homo=("HOMO(eV)", "std"),
            mean_lumo=("LUMO(eV)", "mean"),
            std_lumo=("LUMO(eV)", "std"),
            mean_gap=("Gap(eV)", "mean"),
            std_gap=("Gap(eV)", "std"),
            median_gap=("Gap(eV)", "median"),
        )
        .sort_values("mean_gap")
        .reset_index()
    )
    summary.to_csv(outdir / "summary_by_formulation.csv", index=False)
    return summary


def plot_per_formulation(df: pd.DataFrame, outdir: Path) -> None:
    per_dir = outdir / "per_formulation"
    per_dir.mkdir(parents=True, exist_ok=True)

    folders = sorted(df["Folder"].dropna().unique().tolist())
    for folder in folders:
        dff = df[df["Folder"] == folder].copy()
        if dff.empty:
            continue

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle(f"Formulation Overview: {folder}", fontsize=13)

        cat_counts = dff["Category"].value_counts().sort_index()
        sns.barplot(x=cat_counts.index, y=cat_counts.values, ax=axes[0, 0], color="#4c72b0")
        axes[0, 0].set_title("Category Counts")
        axes[0, 0].set_xlabel("Category")
        axes[0, 0].set_ylabel("Count")
        axes[0, 0].tick_params(axis="x", rotation=30)

        if dff["Gap(eV)"].notna().any():
            sns.boxplot(data=dff, x="Category", y="Gap(eV)", ax=axes[0, 1], color="#55a868")
            axes[0, 1].set_title("Gap by Category")
            axes[0, 1].tick_params(axis="x", rotation=30)
        else:
            axes[0, 1].text(0.5, 0.5, "No valid Gap(eV)", ha="center", va="center")
            axes[0, 1].set_axis_off()

        dff_scatter = dff.dropna(subset=["HOMO(eV)", "LUMO(eV)"])
        if not dff_scatter.empty:
            sns.scatterplot(
                data=dff_scatter,
                x="HOMO(eV)",
                y="LUMO(eV)",
                hue="Category",
                style="Additive",
                ax=axes[1, 0],
                s=45,
            )
            axes[1, 0].set_title("HOMO vs LUMO")
        else:
            axes[1, 0].text(0.5, 0.5, "No valid HOMO/LUMO", ha="center", va="center")
            axes[1, 0].set_axis_off()

        charge_counts = dff["Charge"].dropna().astype(int).value_counts().sort_index()
        if not charge_counts.empty:
            sns.barplot(x=charge_counts.index.astype(str), y=charge_counts.values, ax=axes[1, 1], color="#c44e52")
            axes[1, 1].set_title("Charge Distribution")
            axes[1, 1].set_xlabel("Charge")
            axes[1, 1].set_ylabel("Count")
        else:
            axes[1, 1].text(0.5, 0.5, "No valid Charge", ha="center", va="center")
            axes[1, 1].set_axis_off()

        plt.tight_layout()
        plt.savefig(per_dir / f"{folder}_overview.png", dpi=220)
        plt.close(fig)


def plot_cross_formulation(df: pd.DataFrame, summary: pd.DataFrame, outdir: Path) -> None:
    if summary.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=summary, x="Folder", y="mean_gap", ax=ax, color="#8172b3")
    ax.errorbar(
        x=np.arange(len(summary)),
        y=summary["mean_gap"],
        yerr=summary["std_gap"].fillna(0.0),
        fmt="none",
        ecolor="black",
        elinewidth=1.0,
        capsize=3,
    )
    ax.set_title("Mean Gap by Formulation (with std)")
    ax.set_xlabel("Folder")
    ax.set_ylabel("Mean Gap (eV)")
    ax.tick_params(axis="x", rotation=35)
    plt.tight_layout()
    plt.savefig(outdir / "compare_mean_gap_by_formulation.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 5))
    plot_df = df.dropna(subset=["Gap(eV)"]).copy()
    if not plot_df.empty:
        sns.boxplot(data=plot_df, x="Folder", y="Gap(eV)", ax=ax, color="#64b5cd")
        ax.set_title("Gap Distribution Across Formulations")
        ax.tick_params(axis="x", rotation=35)
        plt.tight_layout()
        plt.savefig(outdir / "compare_gap_distribution_by_formulation.png", dpi=220)
    plt.close(fig)

    ctab = pd.crosstab(df["Folder"], df["Category"])
    if not ctab.empty:
        ctab_pct = ctab.div(ctab.sum(axis=1), axis=0).fillna(0.0)
        fig, ax = plt.subplots(figsize=(11, max(4, 0.6 * len(ctab_pct))))
        sns.heatmap(ctab_pct, cmap="YlGnBu", annot=True, fmt=".2f", cbar_kws={"label": "Fraction"}, ax=ax)
        ax.set_title("Category Composition by Formulation")
        plt.tight_layout()
        plt.savefig(outdir / "compare_category_composition_heatmap.png", dpi=220)
        plt.close(fig)


def write_preliminary_report(df: pd.DataFrame, summary: pd.DataFrame, outdir: Path) -> None:
    report = outdir / "preliminary_analysis.md"
    total_structures = int(df["Filename"].count())
    n_formulations = int(df["Folder"].nunique())

    lines = []
    lines.append("# Preliminary Analysis")
    lines.append("")
    lines.append(f"- Total structures analyzed: **{total_structures}**")
    lines.append(f"- Number of formulations: **{n_formulations}**")
    lines.append("")

    if summary.empty:
        lines.append("No valid rows available for summary statistics.")
    else:
        best = summary.iloc[0]
        worst = summary.iloc[-1]
        lines.append("## Key Findings")
        lines.append(
            f"- Lowest mean gap: **{best['Folder']}** "
            f"(mean gap = {best['mean_gap']:.4f} eV, n = {int(best['n_structures'])})."
        )
        lines.append(
            f"- Highest mean gap: **{worst['Folder']}** "
            f"(mean gap = {worst['mean_gap']:.4f} eV, n = {int(worst['n_structures'])})."
        )
        lines.append("")

        top3 = summary.head(3)
        lines.append("## Top 3 Formulations by Mean Gap (ascending)")
        for _, row in top3.iterrows():
            lines.append(
                f"- {row['Folder']}: mean gap {row['mean_gap']:.4f} eV, "
                f"std {0.0 if pd.isna(row['std_gap']) else row['std_gap']:.4f}, "
                f"n={int(row['n_structures'])}"
            )
        lines.append("")

    cat_counts = df["Category"].value_counts()
    if not cat_counts.empty:
        lines.append("## Global Category Counts")
        for cat, cnt in cat_counts.items():
            lines.append(f"- {cat}: {int(cnt)}")
        lines.append("")

    with open(report, "w") as f:
        f.write("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize and compare HOMO/LUMO results across formulations.")
    parser.add_argument("--csv", default="all_formulations_homolumo.csv", help="Input CSV path.")
    parser.add_argument("--outdir", default="analysis_reports", help="Output directory for plots and report.")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="whitegrid", context="talk")

    try:
        df = load_data(csv_path)
    except Exception as e:
        print(f"[Error] {e}")
        print("No analysis generated.")
        return

    if df.empty:
        print("[Warning] Input CSV is empty. No analysis generated.")
        return

    summary = save_global_summary(df, outdir)
    plot_per_formulation(df, outdir)
    plot_cross_formulation(df, summary, outdir)
    write_preliminary_report(df, summary, outdir)
    print(f"[Done] Analysis outputs written to: {outdir}")


if __name__ == "__main__":
    main()
