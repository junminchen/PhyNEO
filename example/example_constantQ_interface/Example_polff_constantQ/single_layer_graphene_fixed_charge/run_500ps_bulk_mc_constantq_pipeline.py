#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
from pathlib import Path


def run_cmd(cmd: list[str], cwd: Path) -> None:
    print(f"[run] ({cwd}) {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=str(cwd))


def read_density_tail(log_path: Path, n_tail: int = 50) -> tuple[float, float]:
    vals = []
    with log_path.open() as f:
        reader = csv.DictReader(f)
        for r in reader:
            if not r:
                continue
            vals.append(float(r["Density (g/mL)"]))
    tail = vals[-max(1, min(n_tail, len(vals))):]
    avg = sum(tail) / len(tail)
    var = sum((x - avg) ** 2 for x in tail) / len(tail)
    return avg, var ** 0.5


def estimate_sigma_from_log(charge_log: Path, area_nm2: float) -> tuple[float, float]:
    qabs = []
    with charge_log.open() as f:
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue
            sp = line.split()
            if len(sp) < 3:
                continue
            q_c = abs(float(sp[1]))
            qabs.append(q_c)
    if not qabs:
        return 0.0, 0.0
    avg_q = sum(qabs) / len(qabs)
    return avg_q, avg_q / area_nm2


def main() -> None:
    p = argparse.ArgumentParser(description="Pipeline: 500 ps bulk NPT -> MC equil -> production constantQ")
    p.add_argument("--run-root", default="runs/constantq_500ps_mc_prod")
    p.add_argument("--bulk-prod-steps", type=int, default=250000, help="2 fs timestep => 250k steps = 500 ps")
    p.add_argument("--bulk-equil-steps", type=int, default=0)
    p.add_argument("--bulk-report-interval", type=int, default=1000)
    p.add_argument("--prod-equil-steps", type=int, default=50000)
    p.add_argument("--prod-steps", type=int, default=300000)
    p.add_argument("--prod-report-interval", type=int, default=1000)
    p.add_argument("--platform", choices=["CUDA", "CPU", "OpenCL", "Reference"], default="CUDA")
    args = p.parse_args()

    here = Path(__file__).resolve().parent
    root = (here / args.run_root).resolve()
    bulk_src = (here / "../../Example_OPLS/LiPF6_EC_DMC_minimal").resolve()
    intf_src = (here / "../../Example_OPLS/LiPF6_EC_DMC_inert_electrode").resolve()
    opls_root = (here / "../../Example_OPLS").resolve()

    bulk = root / "LiPF6_EC_DMC_minimal"
    intf = root / "LiPF6_EC_DMC_inert_electrode"
    root.mkdir(parents=True, exist_ok=True)
    shutil.copytree(bulk_src, bulk, dirs_exist_ok=True)
    shutil.copytree(intf_src, intf, dirs_exist_ok=True)

    # Rewrite FF paths to absolute paths from the original Example_OPLS tree.
    bulk_cfg_path = bulk / "config.json"
    bulk_cfg = json.loads(bulk_cfg_path.read_text())
    bulk_cfg["forcefield_xml"] = [
        str((opls_root / "opls_salt.xml").resolve()),
        str((opls_root / "opls_solvent.xml").resolve()),
    ]
    bulk_cfg_path.write_text(json.dumps(bulk_cfg, indent=2))

    intf_cfg_path = intf / "config.json"
    intf_cfg = json.loads(intf_cfg_path.read_text())
    intf_cfg["forcefield_xml"] = [
        str((opls_root / "opls_salt.xml").resolve()),
        str((opls_root / "opls_solvent.xml").resolve()),
        str((intf / "electrode_residues.xml").resolve()),
        str((intf / "electrode_ff.xml").resolve()),
    ]
    intf_cfg_path.write_text(json.dumps(intf_cfg, indent=2))

    # Stage 1: bulk NPT 500 ps
    if not (bulk / "start.pdb").exists():
        run_cmd(["conda", "run", "--no-capture-output", "-n", "mpid", "python", "render_packmol.py"], bulk)
        run_cmd(["bash", "-lc", "conda run --no-capture-output -n mpid packmol < packmol.inp"], bulk)
    run_cmd(
        [
            "conda", "run", "--no-capture-output", "-n", "mpid", "python", "run_md_opls.py",
            "--equil-steps", str(args.bulk_equil_steps),
            "--prod-steps", str(args.bulk_prod_steps),
            "--report-interval", str(args.bulk_report_interval),
            "--platform", args.platform,
        ],
        bulk,
    )
    rho_avg, rho_std = read_density_tail(bulk / "npt.log", n_tail=50)

    # Stage 2: build interface + MC equil
    if not (intf / "electrolyte_start.pdb").exists():
        run_cmd(["conda", "run", "--no-capture-output", "-n", "mpid", "python", "render_packmol.py"], intf)
        run_cmd(["bash", "-lc", "conda run --no-capture-output -n mpid packmol < packmol.inp"], intf)
    run_cmd(["conda", "run", "--no-capture-output", "-n", "mpid", "python", "assemble_inert_electrode_system.py"], intf)
    run_cmd(["conda", "run", "--no-capture-output", "-n", "mpid", "python", "mc_gap_equilibrate.py"], intf)

    # Stage 3: production constantQ
    run_cmd(
        [
            "conda", "run", "--no-capture-output", "-n", "mpid", "python", "run_openmm84_inert_electrode.py",
            "--equil-steps", str(args.prod_equil_steps),
            "--prod-steps", str(args.prod_steps),
            "--report-interval", str(args.prod_report_interval),
            "--platform", args.platform,
        ],
        intf,
    )

    cfg = json.loads((intf / "config.json").read_text())
    area_nm2 = float(cfg["cell"]["box_x_angstrom"]) * float(cfg["cell"]["box_y_angstrom"]) / 100.0
    avg_q, sigma_avg = estimate_sigma_from_log(intf / "electrode_charges.log", area_nm2)

    report = root / "pipeline_summary.txt"
    report.write_text(
        "\n".join(
            [
                f"run_root: {root}",
                f"bulk_density_avg_g_ml: {rho_avg:.6f}",
                f"bulk_density_std_g_ml: {rho_std:.6f}",
                f"area_nm2: {area_nm2:.6f}",
                f"constantQ_avg_abs_Qcath_e: {avg_q:.6f}",
                f"constantQ_avg_abs_sigma_e_per_nm2: {sigma_avg:.6f}",
                f"target_sigma_e_per_nm2_note: not directly constrained in this constant-potential run",
            ]
        )
        + "\n"
    )
    print(f"Wrote {report}")
    print("Done.")


if __name__ == "__main__":
    main()
