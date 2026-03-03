#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import datetime as dt
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def ts() -> str:
    return dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(msg: str, flog):
    line = f"[{ts()}] {msg}"
    print(line, flush=True)
    flog.write(line + "\n")
    flog.flush()


def run_cmd(cmd: list[str], cwd: Path, flog):
    log("RUN: " + " ".join(cmd), flog)
    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="", flush=True)
        flog.write(line)
        flog.flush()
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed ({proc.returncode}): {' '.join(cmd)}")


def parse_keyval(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in path.read_text().splitlines():
        s = line.strip()
        if not s or ":" not in s:
            continue
        k, v = s.split(":", 1)
        out[k.strip()] = v.strip()
    return out


def read_profile(csv_path: Path):
    z, li, pf6, ec, dmc, q = [], [], [], [], [], []
    with csv_path.open() as f:
        r = csv.DictReader(f)
        for row in r:
            z.append(float(row["z_angstrom"]))
            li.append(float(row["rho_Li_1_per_nm3"]))
            pf6.append(float(row["rho_PF6_1_per_nm3"]))
            ec.append(float(row["rho_EC_1_per_nm3"]))
            dmc.append(float(row["rho_DMC_1_per_nm3"]))
            q.append(float(row["rho_charge_e_per_nm3"]))
    return np.array(z), np.array(li), np.array(pf6), np.array(ec), np.array(dmc), np.array(q)


def make_comparison_plots(campaign_dir: Path, voltage_dirs: list[tuple[float, Path]]):
    species = [
        ("Li", "rho_Li_1_per_nm3", "Li+"),
        ("PF6", "rho_PF6_1_per_nm3", "PF6-"),
        ("EC", "rho_EC_1_per_nm3", "EC"),
        ("DMC", "rho_DMC_1_per_nm3", "DMC"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=180, sharex=True)
    axes = axes.ravel()

    for (v, vdir) in voltage_dirs:
        p = vdir / "results" / "interface_distribution.csv"
        with p.open() as f:
            rows = list(csv.DictReader(f))
        z = np.array([float(r["z_angstrom"]) for r in rows])
        curves = {
            "rho_Li_1_per_nm3": np.array([float(r["rho_Li_1_per_nm3"]) for r in rows]),
            "rho_PF6_1_per_nm3": np.array([float(r["rho_PF6_1_per_nm3"]) for r in rows]),
            "rho_EC_1_per_nm3": np.array([float(r["rho_EC_1_per_nm3"]) for r in rows]),
            "rho_DMC_1_per_nm3": np.array([float(r["rho_DMC_1_per_nm3"]) for r in rows]),
        }
        for i, (_, key, title) in enumerate(species):
            axes[i].plot(z, curves[key], lw=1.6, label=f"{v:.1f} V")
            axes[i].set_title(title)
            axes[i].set_ylabel("Density (1/nm^3)")
            axes[i].grid(alpha=0.2)

    for ax in axes:
        ax.set_xlabel("z (A)")
    axes[0].legend(frameon=False, fontsize=8, ncol=2)
    fig.suptitle("Interfacial Distribution vs Voltage")
    fig.tight_layout()
    out = campaign_dir / "comparison_distribution_species.png"
    fig.savefig(out)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4.5), dpi=180)
    for (v, vdir) in voltage_dirs:
        p = vdir / "results" / "interface_distribution.csv"
        with p.open() as f:
            rows = list(csv.DictReader(f))
        z = np.array([float(r["z_angstrom"]) for r in rows])
        q = np.array([float(r["rho_charge_e_per_nm3"]) for r in rows])
        ax.plot(z, q, lw=1.6, label=f"{v:.1f} V")
    ax.set_xlabel("z (A)")
    ax.set_ylabel("Charge density (e/nm^3)")
    ax.set_title("Interfacial Charge Density vs Voltage")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, fontsize=8, ncol=3)
    fig.tight_layout()
    out = campaign_dir / "comparison_distribution_charge.png"
    fig.savefig(out)
    plt.close(fig)


def write_report(campaign_dir: Path, summary_rows: list[dict[str, str]], voltages: list[float]):
    csv_out = campaign_dir / "voltage_comparison_summary.csv"
    keys = [
        "voltage_V",
        "samples_used",
        "Q_cath_mean_e",
        "Q_anode_mean_e",
        "Q_total_mean_e",
        "C_interfacial_avg_uF_per_cm2",
        "C_interfacial_avg_block_std_uF_per_cm2",
        "adsorption_Li_cathode_1_per_nm2",
        "adsorption_Li_anode_1_per_nm2",
        "adsorption_PF6_cathode_1_per_nm2",
        "adsorption_PF6_anode_1_per_nm2",
        "charge_adsorption_cathode_e_per_nm2",
        "charge_adsorption_anode_e_per_nm2",
    ]
    with csv_out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for row in summary_rows:
            w.writerow({k: row.get(k, "") for k in keys})

    md_out = campaign_dir / "VOLTAGE_SWEEP_REPORT.md"
    with md_out.open("w") as f:
        f.write("# Voltage Sweep Report (5 points, >=10 ns production each)\n\n")
        f.write("## Setup\n")
        f.write(f"- Voltages (V): {', '.join(f'{v:.1f}' for v in voltages)}\n")
        f.write("- Ensemble: NVT (fixed electrode slabs)\n")
        f.write("- Production per voltage: 10 ns\n")
        f.write("- Equilibration per voltage: 1 ns\n\n")

        f.write("## Key Comparison\n")
        f.write("| Voltage (V) | C_int (uF/cm2) | PF6 cathode (1/nm2) | Li anode (1/nm2) | Q_total mean (e) |\n")
        f.write("|---:|---:|---:|---:|---:|\n")
        for row in summary_rows:
            f.write(
                f"| {row.get('voltage_V','')} | {row.get('C_interfacial_avg_uF_per_cm2','')} | "
                f"{row.get('adsorption_PF6_cathode_1_per_nm2','')} | {row.get('adsorption_Li_anode_1_per_nm2','')} | "
                f"{row.get('Q_total_mean_e','')} |\n"
            )

        f.write("\n## Files\n")
        f.write("- `comparison_distribution_species.png`\n")
        f.write("- `comparison_distribution_charge.png`\n")
        f.write("- `voltage_comparison_summary.csv`\n")
        f.write("- Each voltage directory contains full logs, trajectory, and per-voltage analysis.\n")


def main():
    parser = argparse.ArgumentParser(description="Run 5-voltage sweep with >=10 ns production each and auto-report")
    parser.add_argument("--voltages", default="1,2,3,4,5", help="Comma-separated voltage list in V")
    parser.add_argument("--equil-steps", type=int, default=500000, help="1 ns at 2 fs")
    parser.add_argument("--prod-steps", type=int, default=5000000, help="10 ns at 2 fs")
    parser.add_argument("--report-interval", type=int, default=5000)
    parser.add_argument("--platform", default="CUDA", choices=["CPU", "Reference", "OpenCL", "CUDA"])
    parser.add_argument("--campaign-dir", default="voltage_sweep_5x10ns")
    parser.add_argument("--force", action="store_true", help="Rerun even if DONE.flag exists")
    args = parser.parse_args()

    here = Path(__file__).resolve().parent
    campaign_dir = (here / args.campaign_dir).resolve()
    campaign_dir.mkdir(parents=True, exist_ok=True)
    progress_log = campaign_dir / "progress.log"

    voltages = [float(x.strip()) for x in args.voltages.split(",") if x.strip()]
    if len(voltages) != 5:
        raise ValueError("Please provide exactly 5 voltage points")

    with progress_log.open("a") as flog:
        log("=== Voltage sweep start ===", flog)
        log(f"Voltages: {voltages}", flog)
        log(f"equil_steps={args.equil_steps}, prod_steps={args.prod_steps}, report_interval={args.report_interval}", flog)

        cap_skip = max(1, args.equil_steps // args.report_interval)
        summary_rows: list[dict[str, str]] = []
        voltage_dirs: list[tuple[float, Path]] = []

        for v in voltages:
            tag = f"V{v:.1f}".replace(".", "p")
            vdir = campaign_dir / tag
            vdir.mkdir(parents=True, exist_ok=True)
            done_flag = vdir / "DONE.flag"
            voltage_dirs.append((v, vdir))

            if done_flag.exists() and not args.force:
                log(f"Skip {v:.1f} V (DONE.flag exists)", flog)
            else:
                log(f"Start voltage {v:.1f} V", flog)
                base_pdb = "start_with_electrodes_mc.pdb" if (here / "start_with_electrodes_mc.pdb").exists() else "start_with_electrodes.pdb"
                log(f"Using input PDB: {base_pdb}", flog)
                run_cmd([
                    "conda", "run", "-n", "mpid", "python", "run_openmm84_thick_li_electrode.py",
                    "--pdb", base_pdb,
                    "--equil-steps", str(args.equil_steps),
                    "--prod-steps", str(args.prod_steps),
                    "--report-interval", str(args.report_interval),
                    "--platform", args.platform,
                    "--voltage-v", f"{v}",
                    "--state-log", str((vdir / "npt_thick_li.log").relative_to(here)),
                    "--traj", str((vdir / "traj_thick_li.dcd").relative_to(here)),
                    "--charge-log", str((vdir / "electrode_charges.log").relative_to(here)),
                    "--final-pdb", str((vdir / "final_thick_li.pdb").relative_to(here)),
                ], here, flog)

                run_cmd([
                    "conda", "run", "-n", "mpid", "python", "analyze_interface_distribution.py",
                    "--top", base_pdb,
                    "--traj", str((vdir / "traj_thick_li.dcd").relative_to(here)),
                    "--between-electrodes-only",
                    "--electrode_margin", "1.0",
                    "--interface_width", "5.0",
                    "--nblocks", "5",
                    "--csv", str((vdir / "results" / "interface_distribution.csv").relative_to(here)),
                    "--png", str((vdir / "results" / "interface_distribution.png").relative_to(here)),
                    "--summary", str((vdir / "results" / "interface_distribution_summary.txt").relative_to(here)),
                ], here, flog)

                run_cmd([
                    "conda", "run", "-n", "mpid", "python", "analyze_interfacial_capacitance.py",
                    "--charge-log", str((vdir / "electrode_charges.log").relative_to(here)),
                    "--voltage-v", f"{v}",
                    "--skip", str(cap_skip),
                    "--nblocks", "5",
                    "--summary", str((vdir / "results" / "interfacial_capacitance_summary.txt").relative_to(here)),
                ], here, flog)

                run_cmd([
                    "conda", "run", "-n", "mpid", "python", "visualize_last_frame_electrode_charge.py",
                    "--top", base_pdb,
                    "--traj", str((vdir / "traj_thick_li.dcd").relative_to(here)),
                    "--voltage-v", f"{v}",
                    "--xy-bins", "40",
                    "--png", str((vdir / "results" / "last_frame_electrode_charge_maps.png").relative_to(here)),
                    "--csv", str((vdir / "results" / "last_frame_electrode_atom_charges.csv").relative_to(here)),
                ], here, flog)

                done_flag.write_text(f"done at {ts()}\n")
                log(f"Done voltage {v:.1f} V", flog)

            cap = parse_keyval(vdir / "results" / "interfacial_capacitance_summary.txt")
            dist = parse_keyval(vdir / "results" / "interface_distribution_summary.txt")
            row = {
                "voltage_V": f"{v:.3f}",
                "samples_used": cap.get("samples_used", ""),
                "Q_cath_mean_e": cap.get("Q_cath_mean_e", ""),
                "Q_anode_mean_e": cap.get("Q_anode_mean_e", ""),
                "Q_total_mean_e": cap.get("Q_total_mean_e", ""),
                "C_interfacial_avg_uF_per_cm2": cap.get("C_interfacial_avg_uF_per_cm2", ""),
                "C_interfacial_avg_block_std_uF_per_cm2": cap.get("C_interfacial_avg_block_std_uF_per_cm2", ""),
                "adsorption_Li_cathode_1_per_nm2": dist.get("adsorption_Li_cathode_1_per_nm2", ""),
                "adsorption_Li_anode_1_per_nm2": dist.get("adsorption_Li_anode_1_per_nm2", ""),
                "adsorption_PF6_cathode_1_per_nm2": dist.get("adsorption_PF6_cathode_1_per_nm2", ""),
                "adsorption_PF6_anode_1_per_nm2": dist.get("adsorption_PF6_anode_1_per_nm2", ""),
                "charge_adsorption_cathode_e_per_nm2": dist.get("charge_adsorption_cathode_e_per_nm2", ""),
                "charge_adsorption_anode_e_per_nm2": dist.get("charge_adsorption_anode_e_per_nm2", ""),
            }
            summary_rows.append(row)

        make_comparison_plots(campaign_dir, voltage_dirs)
        write_report(campaign_dir, summary_rows, voltages)
        log("=== Voltage sweep completed ===", flog)
        log(f"Report: {campaign_dir / 'VOLTAGE_SWEEP_REPORT.md'}", flog)


if __name__ == "__main__":
    main()
