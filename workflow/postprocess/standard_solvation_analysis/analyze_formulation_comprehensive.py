import argparse
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def to_numeric_if_exists(df: pd.DataFrame, cols: List[str]) -> None:
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")


def percentile_score(series: pd.Series, higher_better: bool = True) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    out = pd.Series(np.nan, index=s.index, dtype=float)
    valid = s.notna()
    if valid.sum() == 0:
        return out
    ranks = s[valid].rank(method="average", pct=True)
    out.loc[valid] = ranks * 100.0 if higher_better else (1.0 - ranks) * 100.0
    return out


def target_score(series: pd.Series, target: float, tol: float) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    out = pd.Series(np.nan, index=s.index, dtype=float)
    valid = s.notna()
    if valid.sum() == 0:
        return out
    d = np.abs(s[valid] - target) / max(1e-8, tol)
    out.loc[valid] = 100.0 * np.clip(1.0 - d, 0.0, 1.0)
    return out


def ratio_balance_score(series: pd.Series, target: float = 1.0, fold_tol: float = 2.0) -> pd.Series:
    """
    Score ratio closeness on log-scale:
      score=100 at ratio=target
      score=0 at ratio <= target/fold_tol or >= target*fold_tol
    """
    s = pd.to_numeric(series, errors="coerce")
    out = pd.Series(np.nan, index=s.index, dtype=float)
    valid = (s > 0) & s.notna()
    if valid.sum() == 0:
        return out
    logd = np.abs(np.log(s[valid] / target))
    maxd = math.log(max(1.0001, fold_tol))
    out.loc[valid] = 100.0 * np.clip(1.0 - (logd / maxd), 0.0, 1.0)
    return out


def load_shell_summary(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "formulation" not in df.columns:
        raise ValueError("shell summary must contain column: formulation")
    to_numeric_if_exists(
        df,
        [
            "mean_residence_ps",
            "mean_first_shell_residence_ps",
            "mean_second_shell_residence_ps",
            "second_shell_presence_frac",
            "mean_first_cn",
            "mean_second_cn",
            "first_shell_cutoff_A",
            "n_first_shell_events",
            "n_residence_events",
        ],
    )
    return df


def load_homolumo_stats(path: Optional[Path]) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame(columns=["formulation"])
    df = pd.read_csv(path)
    if "Folder" in df.columns and "formulation" not in df.columns:
        df = df.rename(columns={"Folder": "formulation"})
    if "formulation" not in df.columns:
        raise ValueError("HOMO/LUMO CSV must contain 'Folder' or 'formulation'")
    for col in ["Gap(eV)", "LUMO(eV)", "HOMO(eV)"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    grp = df.groupby("formulation", dropna=False)
    out = grp.agg(
        n_homo_samples=("Filename", "count") if "Filename" in df.columns else ("formulation", "count"),
        mean_gap=("Gap(eV)", "mean") if "Gap(eV)" in df.columns else ("formulation", "size"),
        std_gap=("Gap(eV)", "std") if "Gap(eV)" in df.columns else ("formulation", "size"),
        mean_lumo=("LUMO(eV)", "mean") if "LUMO(eV)" in df.columns else ("formulation", "size"),
        mean_homo=("HOMO(eV)", "mean") if "HOMO(eV)" in df.columns else ("formulation", "size"),
    ).reset_index()

    if "Category" in df.columns:
        dcat = df.copy()
        dcat["cat_upper"] = dcat["Category"].astype(str).str.upper()
        ctab = dcat.groupby("formulation")["cat_upper"].value_counts().unstack(fill_value=0)
        ssip_cols = [c for c in ctab.columns if c.startswith("SSIP")]
        cip_cols = [c for c in ctab.columns if c.startswith("CIP")]
        agg_cols = [c for c in ctab.columns if c.startswith("AGG")]
        ctab["ssip_n"] = ctab[ssip_cols].sum(axis=1) if ssip_cols else 0
        ctab["cip_n"] = ctab[cip_cols].sum(axis=1) if cip_cols else 0
        ctab["agg_n"] = ctab[agg_cols].sum(axis=1) if agg_cols else 0
        ctab["cat_total"] = ctab[["ssip_n", "cip_n", "agg_n"]].sum(axis=1).replace(0, np.nan)
        ctab["ssip_frac"] = ctab["ssip_n"] / ctab["cat_total"]
        ctab["cip_frac"] = ctab["cip_n"] / ctab["cat_total"]
        ctab["agg_frac"] = ctab["agg_n"] / ctab["cat_total"]
        ctab["cip_ssip_ratio"] = ctab["cip_n"] / ctab["ssip_n"].replace(0, np.nan)
        out = out.merge(
            ctab[["ssip_n", "cip_n", "agg_n", "ssip_frac", "cip_frac", "agg_frac", "cip_ssip_ratio"]].reset_index(),
            on="formulation",
            how="left",
        )
    return out


def load_rdf_features(rdf_root: Optional[Path]) -> pd.DataFrame:
    if rdf_root is None:
        return pd.DataFrame(columns=["formulation"])
    if not rdf_root.exists():
        return pd.DataFrame(columns=["formulation"])

    files = list(rdf_root.rglob("*_rdf.csv"))
    if len(files) == 0:
        return pd.DataFrame(columns=["formulation"])

    rows = []
    for fp in files:
        try:
            rdf = pd.read_csv(fp)
            if "r_A" not in rdf.columns or "g_r" not in rdf.columns:
                continue
            r = pd.to_numeric(rdf["r_A"], errors="coerce").to_numpy()
            g = pd.to_numeric(rdf["g_r"], errors="coerce").to_numpy()
            valid = np.isfinite(r) & np.isfinite(g)
            r = r[valid]
            g = g[valid]
            if len(r) < 5:
                continue
            mask = r >= 0.5
            if np.any(mask):
                r2 = r[mask]
                g2 = g[mask]
            else:
                r2 = r
                g2 = g
            ip = int(np.argmax(g2))
            peak_r = float(r2[ip])
            peak_g = float(g2[ip])
            area = float(np.trapz(g2, r2))

            name = fp.name
            if name.endswith("_rdf.csv"):
                formulation = name[: -len("_rdf.csv")]
            else:
                formulation = fp.stem
            rows.append(
                {
                    "formulation": formulation,
                    "rdf_peak_r_A": peak_r,
                    "rdf_peak_height": peak_g,
                    "rdf_area": area,
                    "rdf_file": str(fp),
                }
            )
        except Exception:
            continue

    if len(rows) == 0:
        return pd.DataFrame(columns=["formulation"])
    out = pd.DataFrame(rows)
    out = out.drop_duplicates(subset=["formulation"], keep="first")
    return out


def add_scores(
    df: pd.DataFrame,
    gap_objective: str,
    lumo_objective: str,
    target_cn: float,
) -> Tuple[pd.DataFrame, List[Tuple[str, str, float]]]:
    out = df.copy()
    metric_plan: List[Tuple[str, str, float]] = []

    def register(score_col: str, score_series: pd.Series, label: str, weight: float) -> None:
        out[score_col] = score_series
        metric_plan.append((score_col, label, weight))

    tau_col = "mean_first_shell_residence_ps" if "mean_first_shell_residence_ps" in out.columns else "mean_residence_ps"
    if tau_col in out.columns:
        register("score_tau1", percentile_score(out[tau_col], higher_better=True), f"{tau_col} (higher better)", 0.20)
    if "second_shell_presence_frac" in out.columns:
        register(
            "score_second_presence",
            percentile_score(out["second_shell_presence_frac"], higher_better=False),
            "second_shell_presence_frac (lower better)",
            0.10,
        )
    if "mean_second_shell_residence_ps" in out.columns:
        register(
            "score_tau2",
            percentile_score(out["mean_second_shell_residence_ps"], higher_better=False),
            "mean_second_shell_residence_ps (lower better)",
            0.05,
        )

    if "agg_frac" in out.columns:
        register("score_agg", percentile_score(out["agg_frac"], higher_better=False), "agg_frac (lower better)", 0.12)
    if "cip_ssip_ratio" in out.columns:
        register("score_cip_ssip", ratio_balance_score(out["cip_ssip_ratio"], target=1.0, fold_tol=2.0), "cip_ssip_ratio (~1 best)", 0.08)
    if "ssip_frac" in out.columns:
        register("score_ssip", percentile_score(out["ssip_frac"], higher_better=True), "ssip_frac (higher better)", 0.05)

    if "mean_gap" in out.columns:
        register(
            "score_gap",
            percentile_score(out["mean_gap"], higher_better=(gap_objective == "high")),
            f"mean_gap ({gap_objective} better)",
            0.18,
        )
    if "std_gap" in out.columns:
        register("score_gap_std", percentile_score(out["std_gap"], higher_better=False), "std_gap (lower better)", 0.05)
    if "mean_lumo" in out.columns:
        register(
            "score_lumo",
            percentile_score(out["mean_lumo"], higher_better=(lumo_objective == "high")),
            f"mean_lumo ({lumo_objective} better)",
            0.05,
        )

    if "mean_first_cn" in out.columns:
        register(
            "score_cn",
            target_score(out["mean_first_cn"], target=target_cn, tol=1.5),
            f"mean_first_cn (target={target_cn})",
            0.09,
        )
    if "rdf_peak_height" in out.columns:
        register("score_rdf_peak_h", percentile_score(out["rdf_peak_height"], higher_better=True), "rdf_peak_height (higher better)", 0.02)
    if "rdf_peak_r_A" in out.columns:
        register("score_rdf_peak_r", percentile_score(out["rdf_peak_r_A"], higher_better=False), "rdf_peak_r_A (lower better)", 0.01)

    if "n_first_shell_events" in out.columns:
        register(
            "score_n_events",
            percentile_score(out["n_first_shell_events"], higher_better=True),
            "n_first_shell_events (higher better)",
            0.05,
        )
    elif "n_residence_events" in out.columns:
        register(
            "score_n_events",
            percentile_score(out["n_residence_events"], higher_better=True),
            "n_residence_events (higher better)",
            0.05,
        )

    if len(metric_plan) == 0:
        raise ValueError("No usable metrics found for scoring.")

    total_w = float(np.sum([w for _, _, w in metric_plan]))
    out["composite_score"] = 0.0
    for score_col, _, w in metric_plan:
        out["composite_score"] += out[score_col].fillna(50.0) * (w / total_w)
    out = out.sort_values("composite_score", ascending=False).reset_index(drop=True)
    out["rank"] = np.arange(1, len(out) + 1)
    return out, metric_plan


def strengths_and_risks(row: pd.Series) -> Tuple[List[str], List[str]]:
    strengths: List[str] = []
    risks: List[str] = []

    if "score_tau1" in row.index and pd.notna(row["score_tau1"]):
        if row["score_tau1"] >= 65:
            strengths.append("壳层停留时间表现较好")
        elif row["score_tau1"] <= 35:
            risks.append("壳层停留时间偏弱")
    if "score_second_presence" in row.index and pd.notna(row["score_second_presence"]):
        if row["score_second_presence"] >= 65:
            strengths.append("第二壳层占据控制较好")
        elif row["score_second_presence"] <= 35:
            risks.append("第二壳层占据偏高")
    if "score_agg" in row.index and pd.notna(row["score_agg"]):
        if row["score_agg"] >= 65:
            strengths.append("AGG 比例较低")
        elif row["score_agg"] <= 35:
            risks.append("AGG 比例偏高")
    if "score_cip_ssip" in row.index and pd.notna(row["score_cip_ssip"]):
        if row["score_cip_ssip"] <= 35:
            risks.append("CIP/SSIP 比例偏离平衡")
    if "score_gap" in row.index and pd.notna(row["score_gap"]):
        if row["score_gap"] >= 65:
            strengths.append("Gap 指标较优")
        elif row["score_gap"] <= 35:
            risks.append("Gap 指标较弱")
    if "score_cn" in row.index and pd.notna(row["score_cn"]):
        if row["score_cn"] <= 35:
            risks.append("第一壳层配位数偏离目标")
        elif row["score_cn"] >= 65:
            strengths.append("第一壳层配位数接近目标")
    return strengths, risks


def write_report(
    scored: pd.DataFrame,
    metric_plan: List[Tuple[str, str, float]],
    out_md: Path,
    inputs: Dict[str, str],
) -> None:
    lines: List[str] = []
    lines.append("# Comprehensive Formulation Analysis")
    lines.append("")
    lines.append("## Inputs")
    for k, v in inputs.items():
        lines.append(f"- {k}: `{v}`")
    lines.append("")
    lines.append(f"- Formulations analyzed: **{len(scored)}**")
    lines.append("- Composite score range: 0-100 (higher is better).")
    lines.append("")

    lines.append("## Metrics And Weights")
    total_w = float(np.sum([w for _, _, w in metric_plan]))
    for _, label, w in metric_plan:
        lines.append(f"- {label}: {w / total_w:.2f}")
    lines.append("")

    best = scored.iloc[0]
    lines.append("## Best Formulation")
    lines.append(f"- **{best['formulation']}** (score={best['composite_score']:.2f})")
    lines.append("")

    lines.append("## Ranking")
    for _, row in scored.iterrows():
        lines.append(f"- #{int(row['rank'])} {row['formulation']}: score={row['composite_score']:.2f}")
    lines.append("")

    lines.append("## Per-Formulation Evaluation")
    for _, row in scored.iterrows():
        strengths, risks = strengths_and_risks(row)
        lines.append(f"### {row['formulation']}")
        lines.append(f"- Rank: #{int(row['rank'])}, Score: {row['composite_score']:.2f}")
        if "cip_ssip_ratio" in row.index and pd.notna(row["cip_ssip_ratio"]):
            lines.append(f"- CIP/SSIP ratio: {row['cip_ssip_ratio']:.3f}")
        if "mean_first_cn" in row.index and pd.notna(row["mean_first_cn"]):
            lines.append(f"- Mean first-shell CN: {row['mean_first_cn']:.3f}")
        if "mean_gap" in row.index and pd.notna(row["mean_gap"]):
            lines.append(f"- Mean gap (eV): {row['mean_gap']:.4f}")
        if "mean_lumo" in row.index and pd.notna(row["mean_lumo"]):
            lines.append(f"- Mean LUMO (eV): {row['mean_lumo']:.4f}")
        if "rdf_peak_r_A" in row.index and pd.notna(row["rdf_peak_r_A"]):
            lines.append(f"- RDF first peak: r={row['rdf_peak_r_A']:.3f} A, height={row.get('rdf_peak_height', np.nan):.3f}")
        if strengths:
            lines.append(f"- Strengths: {'; '.join(strengths)}")
        if risks:
            lines.append(f"- Risks: {'; '.join(risks)}")
        if not strengths and not risks:
            lines.append("- Notes: 指标完整性不足，建议补充输入文件后再判断。")
        lines.append("")

    out_md.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Comprehensive evaluation of formulations using shell dynamics + RDF + CIP/SSIP + HOMO/LUMO."
    )
    parser.add_argument("--shell-summary", required=True, help="Path to shell_dynamics_summary.csv")
    parser.add_argument("--homolumo-csv", default="", help="Path to merged HOMO/LUMO CSV (optional)")
    parser.add_argument("--rdf-root", default="", help="Root directory containing *_rdf.csv files (optional)")
    parser.add_argument("--outdir", default="analysis_reports", help="Output directory")
    parser.add_argument("--gap-objective", choices=["low", "high"], default="low", help="Whether lower/higher gap is better")
    parser.add_argument("--lumo-objective", choices=["low", "high"], default="high", help="Whether lower/higher LUMO is better")
    parser.add_argument("--target-cn", type=float, default=4.0, help="Target first-shell coordination number")
    args = parser.parse_args()

    shell_path = Path(args.shell_summary)
    if not shell_path.exists():
        raise FileNotFoundError(f"shell summary not found: {shell_path}")
    homolumo_path = Path(args.homolumo_csv) if args.homolumo_csv else None
    rdf_root = Path(args.rdf_root) if args.rdf_root else None

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    shell_df = load_shell_summary(shell_path)
    homo_df = load_homolumo_stats(homolumo_path)
    rdf_df = load_rdf_features(rdf_root)

    merged = shell_df.copy()
    if len(homo_df.columns) > 1:
        merged = merged.merge(homo_df, on="formulation", how="left")
    if len(rdf_df.columns) > 1:
        merged = merged.merge(rdf_df, on="formulation", how="left")

    scored, metric_plan = add_scores(
        merged,
        gap_objective=args.gap_objective,
        lumo_objective=args.lumo_objective,
        target_cn=args.target_cn,
    )

    out_csv = outdir / "comprehensive_formulation_scores.csv"
    out_md = outdir / "comprehensive_formulation_analysis.md"
    scored.to_csv(out_csv, index=False)
    write_report(
        scored,
        metric_plan,
        out_md,
        inputs={
            "shell_summary": str(shell_path),
            "homolumo_csv": str(homolumo_path) if homolumo_path else "N/A",
            "rdf_root": str(rdf_root) if rdf_root else "N/A",
        },
    )

    print(f"[Done] Scores CSV: {out_csv}")
    print(f"[Done] Analysis MD: {out_md}")


if __name__ == "__main__":
    main()
