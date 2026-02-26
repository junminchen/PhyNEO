import argparse
import glob
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import MDAnalysis as mda
from MDAnalysis.analysis.rdf import InterRDF
from MDAnalysis.lib.distances import distance_array

try:
    from scipy.signal import find_peaks, savgol_filter
except Exception:
    find_peaks = None
    savgol_filter = None


def infer_additive_from_topol(folder_dir: Path) -> str:
    topol_file = folder_dir / "topol.top"
    if not topol_file.exists():
        return "UNKNOWN"

    molecules = []
    in_molecules_block = False
    try:
        with open(topol_file, "r") as f:
            for raw_line in f:
                line = raw_line.split(";", 1)[0].strip()
                if not line:
                    continue
                if line.startswith("[") and line.endswith("]"):
                    block = line[1:-1].strip().lower()
                    in_molecules_block = (block == "molecules")
                    continue
                if not in_molecules_block or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) < 2:
                    continue
                try:
                    count = int(parts[1])
                except ValueError:
                    continue
                molecules.append((parts[0], count))
    except Exception:
        return "UNKNOWN"

    a_molecules = [(name, cnt) for name, cnt in molecules if name.startswith("A")]
    if not a_molecules:
        return "NONE"
    a_molecules.sort(key=lambda x: (x[1], x[0]))
    return a_molecules[0][0]


def load_additive_map_from_classify() -> Dict[str, str]:
    try:
        from classify_solvation_env import ADDITIVE_MAP  # type: ignore
        if isinstance(ADDITIVE_MAP, dict):
            return {str(k): str(v) for k, v in ADDITIVE_MAP.items()}
    except Exception:
        pass
    return {}


def parse_additive_map_text(raw: str) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    txt = (raw or "").strip()
    if not txt:
        return mapping
    for item in txt.split(","):
        token = item.strip()
        if not token or ":" not in token:
            continue
        key, val = token.split(":", 1)
        key = key.strip()
        val = val.strip()
        if key:
            mapping[key] = val
    return mapping


def resolve_analysis_selection(
    formulation: str,
    folder_dir: Path,
    u: mda.Universe,
    args: argparse.Namespace,
) -> Tuple[str, str]:
    if args.analysis_target != "additive":
        sel = args.solvent_selection
        if len(u.select_atoms(sel)) == 0:
            raise ValueError(f"manual selection has no atoms: {sel}")
        return sel, "MANUAL"

    additive = args.additive_map_dict.get(formulation)
    if additive is None:
        additive = infer_additive_from_topol(folder_dir)

    if additive in ("NONE", "UNKNOWN", ""):
        if args.skip_no_additive:
            raise ValueError(f"no additive detected for {formulation}")
        fallback = args.additive_fallback_selection
        if len(u.select_atoms(fallback)) == 0:
            raise ValueError(f"fallback selection has no atoms: {fallback}")
        return fallback, additive

    additive_sel = args.additive_selection_template.format(additive=additive)
    if len(u.select_atoms(additive_sel)) == 0:
        if args.skip_no_additive:
            raise ValueError(f"additive selection has no atoms: {additive_sel}")
        fallback = args.additive_fallback_selection
        if len(u.select_atoms(fallback)) == 0:
            raise ValueError(
                f"additive selection empty and fallback has no atoms: {additive_sel} | {fallback}"
            )
        return fallback, additive

    return additive_sel, additive


def discover_systems(
    folder_pattern: str,
    top_patterns: List[str],
    traj_patterns: List[str],
) -> List[Tuple[str, Path, Path]]:
    def find_first_file(base: Path, patterns: List[str]) -> Optional[Path]:
        for pat in patterns:
            pat = pat.strip()
            if not pat:
                continue
            hits = sorted(base.glob(pat))
            if hits:
                return hits[0]
            # fallback recursive search for nested trajectory folders
            hits_r = sorted(base.rglob(pat))
            if hits_r:
                return hits_r[0]
        return None

    systems = []
    for folder in sorted(Path(".").glob(folder_pattern)):
        if not folder.is_dir():
            continue

        top_file = find_first_file(folder, top_patterns)
        traj_file = find_first_file(folder, traj_patterns)

        if top_file is None or traj_file is None:
            print(
                f"[Skip] {folder.name}: topology/trajectory not found "
                f"(top_patterns={top_patterns}, traj_patterns={traj_patterns})"
            )
            continue

        systems.append((folder.name, top_file, traj_file))
    return systems


def parse_manifest(manifest_csv: Path) -> List[Tuple[str, Path, Path]]:
    df = pd.read_csv(manifest_csv)
    required = {"formulation", "topology", "trajectory"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"manifest missing columns: {sorted(missing)}")

    systems = []
    for _, row in df.iterrows():
        name = str(row["formulation"])
        top = Path(str(row["topology"]))
        traj = Path(str(row["trajectory"]))
        systems.append((name, top, traj))
    return systems


def smooth_rdf(g: np.ndarray) -> np.ndarray:
    if savgol_filter is None or len(g) < 7:
        return g
    win = min(21, len(g) if len(g) % 2 == 1 else len(g) - 1)
    if win < 5:
        return g
    try:
        return savgol_filter(g, window_length=win, polyorder=3)
    except Exception:
        return g


def find_shell_cutoffs(r: np.ndarray, g: np.ndarray) -> Tuple[float, float]:
    g_s = smooth_rdf(g)

    if find_peaks is None:
        i_peak = int(np.argmax(g_s))
        i_min = i_peak + int(np.argmin(g_s[i_peak:])) if i_peak < len(g_s) - 1 else len(g_s) - 1
        r1 = float(r[i_min])
        r2 = min(float(r[-1]), 1.8 * r1)
        return r1, r2

    peaks, _ = find_peaks(g_s)
    mins, _ = find_peaks(-g_s)

    if len(peaks) == 0:
        return float(r[len(r) // 4]), float(r[len(r) // 2])

    p1 = int(peaks[0])
    mins_after_p1 = mins[mins > p1]
    if len(mins_after_p1) == 0:
        r1 = float(r[min(len(r) - 1, p1 + max(1, len(r) // 20))])
        r2 = min(float(r[-1]), 1.8 * r1)
        return r1, r2

    m1 = int(mins_after_p1[0])
    r1 = float(r[m1])

    p2_candidates = peaks[peaks > m1]
    if len(p2_candidates) == 0:
        r2 = min(float(r[-1]), 1.8 * r1)
        return r1, r2

    p2 = int(p2_candidates[0])
    mins_after_p2 = mins[mins > p2]
    if len(mins_after_p2) == 0:
        r2 = min(float(r[-1]), 1.8 * r1)
    else:
        r2 = float(r[int(mins_after_p2[0])])

    if r2 <= r1:
        r2 = min(float(r[-1]), 1.8 * r1)
    return r1, r2


def compute_rdf_and_shells(
    u: mda.Universe,
    cation_sel: str,
    solvent_sel: str,
    start: int,
    stop: Optional[int],
    step: int,
    rdf_nbins: int,
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    cations = u.select_atoms(cation_sel)
    solvent_atoms = u.select_atoms(solvent_sel)
    if len(cations) == 0:
        raise ValueError(f"No atoms in cation selection: {cation_sel}")
    if len(solvent_atoms) == 0:
        raise ValueError(f"No atoms in solvent selection: {solvent_sel}")

    max_r = min(u.dimensions[:3]) / 2.0
    rdf = InterRDF(cations, solvent_atoms, nbins=rdf_nbins, range=(0.0, max_r))
    rdf.run(start=start, stop=stop, step=step)

    r = np.asarray(rdf.results.bins)
    g = np.asarray(rdf.results.rdf)
    r1, r2 = find_shell_cutoffs(r, g)
    return r, g, r1, r2


def analyze_shell_dynamics(
    u: mda.Universe,
    cation_sel: str,
    solvent_sel: str,
    r1: float,
    r2: float,
    start: int,
    stop: Optional[int],
    step: int,
    dt_ps: Optional[float],
) -> Dict[str, np.ndarray]:
    cations = u.select_atoms(cation_sel)
    solvent = u.select_atoms(solvent_sel)

    cat_pos = cations.positions
    sol_pos = solvent.positions
    if cat_pos.shape[0] == 0 or sol_pos.shape[0] == 0:
        raise ValueError("Empty selected atoms for dynamics analysis.")

    sol_resindex = solvent.resindices.astype(int)
    unique_res = np.unique(sol_resindex)

    active_contacts: Dict[Tuple[int, int], int] = {}
    durations_frames: List[int] = []
    first_cn_per_cat_frame: List[int] = []
    second_cn_per_cat_frame: List[int] = []
    second_shell_presence: List[int] = []

    frame_ids = []
    for ts in u.trajectory[start:stop:step]:
        frame_ids.append(ts.frame)
        dist = distance_array(cations.positions, solvent.positions, box=ts.dimensions)
        current_contacts = set()

        for ci in range(dist.shape[0]):
            drow = dist[ci]
            first_res = np.unique(sol_resindex[drow <= r1])
            second_res = np.unique(sol_resindex[(drow > r1) & (drow <= r2)])

            first_cn_per_cat_frame.append(int(len(first_res)))
            second_cn_per_cat_frame.append(int(len(second_res)))
            second_shell_presence.append(1 if len(second_res) > 0 else 0)

            for resid in first_res:
                current_contacts.add((ci, int(resid)))

        ended = set(active_contacts.keys()) - current_contacts
        for pair in ended:
            start_frame_idx = active_contacts.pop(pair)
            durations_frames.append(len(frame_ids) - 1 - start_frame_idx)

        for pair in current_contacts:
            if pair not in active_contacts:
                active_contacts[pair] = len(frame_ids) - 1

    last_idx = len(frame_ids) - 1
    for pair, start_idx in active_contacts.items():
        durations_frames.append(last_idx - start_idx + 1)

    if dt_ps is None:
        try:
            base_dt = float(u.trajectory.dt)
        except Exception:
            base_dt = 1.0
        dt_ps_eff = base_dt * step
    else:
        dt_ps_eff = float(dt_ps) * step

    durations_ps = np.asarray(durations_frames, dtype=float) * dt_ps_eff

    return {
        "durations_ps": durations_ps,
        "first_cn": np.asarray(first_cn_per_cat_frame, dtype=float),
        "second_cn": np.asarray(second_cn_per_cat_frame, dtype=float),
        "second_presence": np.asarray(second_shell_presence, dtype=float),
        "n_frames": np.asarray([len(frame_ids)], dtype=int),
        "n_cations": np.asarray([len(cations)], dtype=int),
        "n_solvent_residues": np.asarray([len(unique_res)], dtype=int),
    }


def save_formulation_plots(
    outdir: Path,
    name: str,
    r: np.ndarray,
    g: np.ndarray,
    r1: float,
    r2: float,
    durations_ps: np.ndarray,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(r, g, lw=1.8, color="#2a9d8f", label="RDF")
    ax.axvline(r1, color="#e76f51", ls="--", lw=1.5, label=f"1st shell cutoff ({r1:.2f} A)")
    ax.axvline(r2, color="#264653", ls="--", lw=1.5, label=f"2nd shell cutoff ({r2:.2f} A)")
    ax.fill_between(r, 0, g, where=(r <= r1), color="#e76f51", alpha=0.12)
    ax.fill_between(r, 0, g, where=((r > r1) & (r <= r2)), color="#264653", alpha=0.10)
    ax.set_xlabel("r (A)")
    ax.set_ylabel("g(r)")
    ax.set_title(f"RDF and Shell Cutoffs: {name}")
    ax.legend()
    plt.tight_layout()
    plt.savefig(outdir / f"{name}_rdf_shells.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    if len(durations_ps) > 0:
        sns.histplot(durations_ps, bins=min(60, max(10, len(durations_ps) // 5)), kde=True, ax=ax, color="#4c78a8")
    ax.set_xlabel("First-shell residence duration (ps)")
    ax.set_ylabel("Count")
    ax.set_title(f"Residence Time Distribution: {name}")
    plt.tight_layout()
    plt.savefig(outdir / f"{name}_residence_hist.png", dpi=220)
    plt.close(fig)


def save_compare_plots(summary_df: pd.DataFrame, events_df: pd.DataFrame, outdir: Path) -> None:
    if summary_df.empty:
        return

    sort_names = summary_df.sort_values("mean_residence_ps", ascending=False)["formulation"].tolist()

    fig, ax = plt.subplots(figsize=(9, 4.5))
    sns.barplot(data=summary_df, x="formulation", y="mean_residence_ps", order=sort_names, ax=ax, color="#1d3557")
    ax.errorbar(
        x=np.arange(len(summary_df)),
        y=summary_df.set_index("formulation").loc[sort_names, "mean_residence_ps"].values,
        yerr=summary_df.set_index("formulation").loc[sort_names, "std_residence_ps"].fillna(0.0).values,
        fmt="none",
        ecolor="black",
        capsize=3,
    )
    ax.set_title("First-Shell Mean Residence Time by Formulation")
    ax.set_xlabel("Formulation")
    ax.set_ylabel("Residence time (ps)")
    ax.tick_params(axis="x", rotation=30)
    plt.tight_layout()
    plt.savefig(outdir / "compare_mean_residence_time.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    sns.barplot(data=summary_df, x="formulation", y="second_shell_presence_frac", order=sort_names, ax=ax, color="#e76f51")
    ax.set_title("Second-Shell Presence Fraction by Formulation")
    ax.set_xlabel("Formulation")
    ax.set_ylabel("Fraction of cation-frames in 2nd shell")
    ax.tick_params(axis="x", rotation=30)
    plt.tight_layout()
    plt.savefig(outdir / "compare_second_shell_presence.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    melted = summary_df.melt(
        id_vars=["formulation"],
        value_vars=["mean_first_cn", "mean_second_cn"],
        var_name="shell_metric",
        value_name="coordination_number",
    )
    sns.barplot(data=melted, x="formulation", y="coordination_number", hue="shell_metric", ax=ax)
    ax.set_title("Average Coordination Number by Formulation")
    ax.set_xlabel("Formulation")
    ax.set_ylabel("CN")
    ax.tick_params(axis="x", rotation=30)
    plt.tight_layout()
    plt.savefig(outdir / "compare_coordination_numbers.png", dpi=220)
    plt.close(fig)

    if not events_df.empty:
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.violinplot(data=events_df, x="formulation", y="residence_ps", cut=0, inner="quartile", ax=ax)
        ax.set_title("First-Shell Residence Time Distribution Across Formulations")
        ax.set_xlabel("Formulation")
        ax.set_ylabel("Residence time (ps)")
        ax.tick_params(axis="x", rotation=30)
        plt.tight_layout()
        plt.savefig(outdir / "compare_residence_time_violin.png", dpi=220)
        plt.close(fig)


def run_analysis(args: argparse.Namespace) -> None:
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    per_dir = outdir / "per_formulation"
    per_dir.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="whitegrid", context="talk")
    args.additive_map_dict = load_additive_map_from_classify()
    args.additive_map_dict.update(parse_additive_map_text(args.additive_map))

    if args.manifest:
        systems = parse_manifest(Path(args.manifest))
    else:
        systems = discover_systems(
            folder_pattern=args.folder_pattern,
            top_patterns=args.top_patterns.split(","),
            traj_patterns=args.traj_patterns.split(","),
        )

    if len(systems) == 0:
        print("[Error] No valid systems found.")
        return

    summary_rows = []
    all_events = []

    for name, top, traj in systems:
        print(f"[Run ] {name}")
        if not top.exists() or not traj.exists():
            print(f"[Skip] {name}: missing files ({top}, {traj})")
            continue

        try:
            u = mda.Universe(str(top), str(traj))
            folder_dir = Path(top).resolve().parent
            target_sel, additive_name = resolve_analysis_selection(name, folder_dir, u, args)
            print(f"[Info] {name}: target={args.analysis_target} additive={additive_name} selection={target_sel}")
            r, g, r1, r2 = compute_rdf_and_shells(
                u=u,
                cation_sel=args.cation_selection,
                solvent_sel=target_sel,
                start=args.start,
                stop=args.stop,
                step=args.step,
                rdf_nbins=args.rdf_nbins,
            )
            dyn = analyze_shell_dynamics(
                u=u,
                cation_sel=args.cation_selection,
                solvent_sel=target_sel,
                r1=r1,
                r2=r2,
                start=args.start,
                stop=args.stop,
                step=args.step,
                dt_ps=args.dt_ps,
            )
        except Exception as e:
            print(f"[Fail] {name}: {e}")
            continue

        durations_ps = dyn["durations_ps"]
        mean_res = float(np.mean(durations_ps)) if len(durations_ps) > 0 else 0.0
        med_res = float(np.median(durations_ps)) if len(durations_ps) > 0 else 0.0
        p90_res = float(np.percentile(durations_ps, 90)) if len(durations_ps) > 0 else 0.0
        std_res = float(np.std(durations_ps)) if len(durations_ps) > 0 else 0.0
        mean_first_cn = float(np.mean(dyn["first_cn"])) if len(dyn["first_cn"]) > 0 else 0.0
        mean_second_cn = float(np.mean(dyn["second_cn"])) if len(dyn["second_cn"]) > 0 else 0.0
        second_presence = float(np.mean(dyn["second_presence"])) if len(dyn["second_presence"]) > 0 else 0.0

        summary_rows.append(
            {
                "formulation": name,
                "analysis_target": args.analysis_target,
                "additive": additive_name,
                "target_selection": target_sel,
                "topology": str(top),
                "trajectory": str(traj),
                "first_shell_cutoff_A": r1,
                "second_shell_cutoff_A": r2,
                "n_residence_events": int(len(durations_ps)),
                "mean_residence_ps": mean_res,
                "median_residence_ps": med_res,
                "p90_residence_ps": p90_res,
                "std_residence_ps": std_res,
                "mean_first_cn": mean_first_cn,
                "mean_second_cn": mean_second_cn,
                "second_shell_presence_frac": second_presence,
                "n_frames_used": int(dyn["n_frames"][0]),
                "n_cations": int(dyn["n_cations"][0]),
            }
        )

        if len(durations_ps) > 0:
            all_events.append(pd.DataFrame({"formulation": name, "residence_ps": durations_ps}))

        pd.DataFrame({"formulation": name, "residence_ps": durations_ps}).to_csv(
            per_dir / f"{name}_residence_events.csv", index=False
        )
        pd.DataFrame({"r_A": r, "g_r": g}).to_csv(per_dir / f"{name}_rdf.csv", index=False)
        save_formulation_plots(per_dir, name, r, g, r1, r2, durations_ps)
        print(f"[Done] {name}: r1={r1:.3f} A r2={r2:.3f} A mean_tau={mean_res:.3f} ps")

    if len(summary_rows) == 0:
        print("[Error] No system completed successfully.")
        return

    summary_df = pd.DataFrame(summary_rows).sort_values("mean_residence_ps", ascending=False).reset_index(drop=True)
    summary_df.to_csv(outdir / "shell_dynamics_summary.csv", index=False)

    events_df = pd.concat(all_events, ignore_index=True) if all_events else pd.DataFrame(columns=["formulation", "residence_ps"])
    events_df.to_csv(outdir / "all_residence_events.csv", index=False)
    save_compare_plots(summary_df, events_df, outdir)

    md_lines = [
        "# Solvation Shell Dynamics Report",
        "",
        f"- Systems analyzed: **{len(summary_df)}**",
        f"- Analysis target: `{args.analysis_target}`",
        f"- Cation selection: `{args.cation_selection}`",
        (
            f"- Target selection strategy: additive template "
            f"`{args.additive_selection_template}` + fallback `{args.additive_fallback_selection}`"
            if args.analysis_target == "additive"
            else f"- Target selection (manual): `{args.solvent_selection}`"
        ),
        "",
        "## Ranking by Mean First-Shell Residence Time",
    ]
    for _, row in summary_df.iterrows():
        md_lines.append(
            f"- {row['formulation']}: mean={row['mean_residence_ps']:.3f} ps, "
            f"median={row['median_residence_ps']:.3f} ps, "
            f"2nd-shell frac={row['second_shell_presence_frac']:.3f}"
        )
    (outdir / "shell_dynamics_report.md").write_text("\n".join(md_lines) + "\n")
    print(f"[Done] outputs written to: {outdir}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Analyze first-shell residence dynamics and second-shell occupation across formulations."
    )
    p.add_argument("--manifest", default="", help="CSV with columns: formulation,topology,trajectory")
    p.add_argument("--folder-pattern", default="newer*", help="Folder glob pattern for auto-discovery.")
    p.add_argument(
        "--top-patterns",
        default="solvent_salt.pdb,*.pdb,*.gro,*.prmtop,*.psf",
        help="Comma-separated topology patterns.",
    )
    p.add_argument(
        "--traj-patterns",
        default="transport_results/nvt.dcd,*.dcd,*.xtc,*.nc,*.trr",
        help="Comma-separated trajectory patterns.",
    )
    p.add_argument(
        "--analysis-target",
        choices=["additive", "manual"],
        default="additive",
        help="Analyze additive shell dynamics (default) or use manual selection.",
    )
    p.add_argument("--cation-selection", default="resname LI", help="MDAnalysis selection for cations.")
    p.add_argument(
        "--solvent-selection",
        default="(resname EC EMC DMC FEC DEC PC) and (name O* or type O*)",
        help="Manual target selection when --analysis-target manual.",
    )
    p.add_argument(
        "--additive-selection-template",
        default="resname {additive} and (name O* or type O*)",
        help="Selection template for additive mode. '{additive}' will be replaced by detected residue name.",
    )
    p.add_argument(
        "--additive-fallback-selection",
        default="(resname EC EMC DMC FEC DEC PC) and (name O* or type O*)",
        help="Fallback selection when additive is NONE/UNKNOWN or selection empty.",
    )
    p.add_argument(
        "--additive-map",
        default="",
        help="Optional override map: 'folder1:FEC,folder2:VC'.",
    )
    p.add_argument(
        "--skip-no-additive",
        action="store_true",
        help="In additive mode, skip formulations without detected additive.",
    )
    p.add_argument("--start", type=int, default=0, help="Start frame index.")
    p.add_argument("--stop", type=int, default=None, help="Stop frame index (exclusive).")
    p.add_argument("--step", type=int, default=1, help="Frame stride.")
    p.add_argument("--dt-ps", type=float, default=None, help="Override frame dt in ps.")
    p.add_argument("--rdf-nbins", type=int, default=250, help="RDF bins.")
    p.add_argument("--outdir", default="shell_dynamics_reports", help="Output directory.")
    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run_analysis(args)


if __name__ == "__main__":
    main()
