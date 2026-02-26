import argparse
from pathlib import Path
from typing import Dict, List, Tuple

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
    if higher_better:
        out.loc[valid] = ranks * 100.0
    else:
        out.loc[valid] = (1.0 - ranks) * 100.0
    return out


def build_metric_plan(df: pd.DataFrame) -> List[Tuple[str, str, float, bool]]:
    # (score_col, source_col, weight, higher_better)
    plan: List[Tuple[str, str, float, bool]] = []

    if "mean_first_shell_residence_ps" in df.columns:
        plan.append(("score_tau1", "mean_first_shell_residence_ps", 0.35, True))
    elif "mean_residence_ps" in df.columns:
        plan.append(("score_tau1", "mean_residence_ps", 0.35, True))

    if "second_shell_presence_frac" in df.columns:
        plan.append(("score_second_presence", "second_shell_presence_frac", 0.25, False))

    if "mean_second_shell_residence_ps" in df.columns:
        plan.append(("score_tau2", "mean_second_shell_residence_ps", 0.15, False))

    if "state_fraction_second_shell" in df.columns:
        plan.append(("score_state_second", "state_fraction_second_shell", 0.15, False))

    if "n_first_shell_events" in df.columns:
        plan.append(("score_sample", "n_first_shell_events", 0.10, True))
    elif "n_residence_events" in df.columns:
        plan.append(("score_sample", "n_residence_events", 0.10, True))

    return plan


def compute_scores(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[Tuple[str, str, float, bool]]]:
    out = df.copy()
    metric_plan = build_metric_plan(out)
    if len(metric_plan) == 0:
        raise ValueError("No usable metric columns found for scoring.")

    used_weights = []
    for score_col, source_col, weight, higher_better in metric_plan:
        out[score_col] = percentile_score(out[source_col], higher_better=higher_better)
        used_weights.append(weight)

    total_weight = float(np.sum(used_weights))
    if total_weight <= 0:
        raise ValueError("Invalid metric weights.")

    out["composite_score"] = 0.0
    for score_col, _, weight, _ in metric_plan:
        out["composite_score"] += out[score_col].fillna(50.0) * (weight / total_weight)

    out = out.sort_values("composite_score", ascending=False).reset_index(drop=True)
    out["rank"] = np.arange(1, len(out) + 1)
    return out, metric_plan


def metric_label(source_col: str, higher_better: bool) -> str:
    direction = "higher is better" if higher_better else "lower is better"
    return f"{source_col} ({direction})"


def strengths_and_risks(row: pd.Series, scored_df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    strengths: List[str] = []
    risks: List[str] = []

    if "mean_first_shell_residence_ps" in row.index and pd.notna(row["mean_first_shell_residence_ps"]):
        median_tau1 = scored_df["mean_first_shell_residence_ps"].median()
        if row["mean_first_shell_residence_ps"] >= median_tau1:
            strengths.append("第一壳层停留时间不低于中位数")
        else:
            risks.append("第一壳层停留时间偏低")
    elif "mean_residence_ps" in row.index and pd.notna(row["mean_residence_ps"]):
        median_tau = scored_df["mean_residence_ps"].median()
        if row["mean_residence_ps"] >= median_tau:
            strengths.append("平均停留时间不低于中位数")
        else:
            risks.append("平均停留时间偏低")

    if "second_shell_presence_frac" in row.index and pd.notna(row["second_shell_presence_frac"]):
        median_second = scored_df["second_shell_presence_frac"].median()
        if row["second_shell_presence_frac"] <= median_second:
            strengths.append("第二壳层占据率较低")
        else:
            risks.append("第二壳层占据率偏高")

    if "n_first_shell_events" in row.index and pd.notna(row["n_first_shell_events"]):
        median_evt = scored_df["n_first_shell_events"].median()
        if row["n_first_shell_events"] >= median_evt:
            strengths.append("第一壳层事件数充足")
        else:
            risks.append("第一壳层事件数偏少，统计稳健性较弱")
    elif "n_residence_events" in row.index and pd.notna(row["n_residence_events"]):
        median_evt = scored_df["n_residence_events"].median()
        if row["n_residence_events"] >= median_evt:
            strengths.append("停留事件数充足")
        else:
            risks.append("停留事件数偏少，统计稳健性较弱")

    return strengths, risks


def write_report(
    scored_df: pd.DataFrame,
    metric_plan: List[Tuple[str, str, float, bool]],
    out_md: Path,
) -> None:
    lines: List[str] = []
    lines.append("# Shell Dynamics Summary Analysis")
    lines.append("")
    lines.append(f"- Formulations analyzed: **{len(scored_df)}**")
    lines.append("- Ranking basis: weighted percentile score (`composite_score`, 0-100).")
    lines.append("")
    lines.append("## Metrics Used")
    total_weight = float(np.sum([w for _, _, w, _ in metric_plan]))
    for _, source_col, weight, higher_better in metric_plan:
        lines.append(f"- `{source_col}`: weight={weight / total_weight:.2f}, {metric_label(source_col, higher_better)}")
    lines.append("")

    top = scored_df.iloc[0]
    lines.append("## Best Formulation")
    lines.append(f"- **{top['formulation']}** (composite_score={top['composite_score']:.2f})")
    lines.append("")

    lines.append("## Ranking")
    for _, row in scored_df.iterrows():
        lines.append(f"- #{int(row['rank'])} {row['formulation']}: composite_score={row['composite_score']:.2f}")
    lines.append("")

    lines.append("## Per-Formulation Notes")
    for _, row in scored_df.iterrows():
        strengths, risks = strengths_and_risks(row, scored_df)
        lines.append(f"### {row['formulation']}")
        lines.append(f"- Rank: #{int(row['rank'])}, Score: {row['composite_score']:.2f}")
        if strengths:
            lines.append(f"- Strengths: {'; '.join(strengths)}")
        if risks:
            lines.append(f"- Risks: {'; '.join(risks)}")
        if not strengths and not risks:
            lines.append("- Notes: 无足够字段进行优劣判断。")
        lines.append("")

    out_md.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze shell_dynamics_summary.csv and generate ranking report.")
    parser.add_argument("--csv", required=True, help="Path to shell_dynamics_summary.csv")
    parser.add_argument("--outdir", default="shell_dynamics_reports", help="Output directory")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if "formulation" not in df.columns:
        raise ValueError("CSV must contain column: formulation")
    if df.empty:
        raise ValueError("Input CSV is empty")

    to_numeric_if_exists(
        df,
        [
            "mean_residence_ps",
            "mean_first_shell_residence_ps",
            "mean_second_shell_residence_ps",
            "second_shell_presence_frac",
            "state_fraction_second_shell",
            "n_first_shell_events",
            "n_residence_events",
        ],
    )

    scored_df, metric_plan = compute_scores(df)
    scored_csv = outdir / "shell_dynamics_scored.csv"
    report_md = outdir / "shell_dynamics_summary_analysis.md"

    scored_df.to_csv(scored_csv, index=False)
    write_report(scored_df, metric_plan, report_md)

    print(f"[Done] Scored table: {scored_csv}")
    print(f"[Done] Analysis report: {report_md}")


if __name__ == "__main__":
    main()
