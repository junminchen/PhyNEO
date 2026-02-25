"""Minimal PhyNEO OpenMM protocol with validated system loading and MPID scales.

This module intentionally keeps only commonly used pieces:
- validate_inputs
- apply_mpid_scale_exclusions
- load_phyneo_system
- run_protocol (minimal staged MD runner)
"""

from __future__ import annotations

import argparse
import copy
import importlib
import importlib.util
import json
import os
import sys
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Keep full backward-compatible protocol APIs from the original toolkit module.
def _load_legacy_protocol_symbols():
    """
    Try loading legacy toolkit/protocol.py symbols without forcing toolkit/__init__.py.
    Returns (symbols_dict, error_message_or_none).
    """
    protocol_path = Path(__file__).resolve().parent / "toolkit" / "protocol.py"
    if not protocol_path.exists():
        return {}, f"Legacy protocol not found: {protocol_path}"
    try:
        spec = importlib.util.spec_from_file_location("phyneo_openmm_legacy_protocol", str(protocol_path))
        if spec is None or spec.loader is None:
            return {}, "Unable to create module spec for legacy protocol."
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        keys = [
            "ComponentType",
            "Component",
            "predict_density",
            "search_mixture",
            "predict_box",
            "load_topo",
            "generate_system_gro",
            "write_gro",
            "Protocol",
            "DensityProtocol",
            "TransportProtocol",
            "HVapProtocol",
        ]
        return {k: getattr(module, k) for k in keys if hasattr(module, k)}, None
    except Exception as exc:
        return {}, str(exc)


_LEGACY_SYMBOLS, _LEGACY_IMPORT_ERROR = _load_legacy_protocol_symbols()
ComponentType = _LEGACY_SYMBOLS.get("ComponentType")
Component = _LEGACY_SYMBOLS.get("Component")
predict_density = _LEGACY_SYMBOLS.get("predict_density")
search_mixture = _LEGACY_SYMBOLS.get("search_mixture")
predict_box = _LEGACY_SYMBOLS.get("predict_box")
load_topo = _LEGACY_SYMBOLS.get("load_topo")
generate_system_gro = _LEGACY_SYMBOLS.get("generate_system_gro")
write_gro = _LEGACY_SYMBOLS.get("write_gro")
Protocol = _LEGACY_SYMBOLS.get("Protocol")
DensityProtocol = _LEGACY_SYMBOLS.get("DensityProtocol")
TransportProtocol = _LEGACY_SYMBOLS.get("TransportProtocol")
HVapProtocol = _LEGACY_SYMBOLS.get("HVapProtocol")


DEFAULT_M_SCALES = [0.0, 0.0, 0.0, 0.0, 0.0]  # 12,13,14,15,16
DEFAULT_P_SCALES = [0.0, 0.0, 0.0, 0.0, 0.0]  # 12,13,14,15,16
DEFAULT_D_SCALES = [1.0, 1.0, 1.0, 1.0, 1.0]  # recorded only


def import_openmm_modules():
    """Import site-packages openmm, avoiding local package shadowing."""
    root = Path(__file__).resolve().parents[1]
    try:
        import openmm as mm
        import openmm.app as app
        import openmm.unit as unit
        local_pkg = root / "openmm"
        if str(local_pkg) in str(getattr(mm, "__file__", "")):
            raise ImportError("shadowed by local workspace package")
        return mm, app, unit
    except Exception:
        prune = {"", str(root), str(Path.cwd()), os.getcwd()}
        sys.path = [p for p in sys.path if p not in prune]
        sys.modules.pop("openmm", None)
        try:
            mm = importlib.import_module("openmm")
            app = importlib.import_module("openmm.app")
            unit = importlib.import_module("openmm.unit")
            return mm, app, unit
        except Exception:
            mm = importlib.import_module("simtk.openmm")
            app = importlib.import_module("simtk.openmm.app")
            unit = importlib.import_module("simtk.unit")
            return mm, app, unit


def _load_plugin_dir(mm):
    plugin_dir = os.environ.get("OPENMM_PLUGIN_DIR")
    if plugin_dir:
        mm.Platform.loadPluginsFromDirectory(plugin_dir)
    return plugin_dir


def _vprint(verbose, msg):
    if verbose:
        print(f"[phyneo_protocol] {msg}")


def _resolve_constraints(app, constraints):
    if constraints is None:
        return None
    text = str(constraints).strip().lower()
    if text in {"none", "false", "off"}:
        return None
    if text == "hbonds":
        return app.HBonds
    if text == "allbonds":
        return app.AllBonds
    if text == "hangles":
        return app.HAngles
    raise ValueError(f"Unsupported constraints: {constraints}")


def _bonded_shells(topology, max_depth=5):
    atoms = list(topology.atoms())
    n = len(atoms)
    adj = [set() for _ in range(n)]
    for bond in topology.bonds():
        i = bond[0].index
        j = bond[1].index
        adj[i].add(j)
        adj[j].add(i)

    shells_all = []
    for i in range(n):
        visited = {i}
        frontier = {i}
        shells = {d: set() for d in range(1, max_depth + 1)}
        for d in range(1, max_depth + 1):
            nxt = set()
            for u in frontier:
                nxt |= (adj[u] - visited)
            shells[d] = nxt
            visited |= nxt
            frontier = nxt
        shells_all.append(shells)
    return shells_all


def validate_inputs(
    pdb_path,
    xml_path,
    nonbonded_method="NoCutoff",
    platform="CPU",
    m_scales=None,
    p_scales=None,
    d_scales=None,
):
    """Validate user inputs before system loading."""
    pdb = Path(pdb_path).expanduser().resolve()
    xml = Path(xml_path).expanduser().resolve()
    if not pdb.exists():
        raise FileNotFoundError(f"PDB file not found: {pdb}")
    if not xml.exists():
        raise FileNotFoundError(f"XML file not found: {xml}")
    if pdb.suffix.lower() != ".pdb":
        raise ValueError(f"Expected .pdb file: {pdb}")
    if xml.suffix.lower() != ".xml":
        raise ValueError(f"Expected .xml file: {xml}")

    method = str(nonbonded_method).strip().upper()
    if method not in {"PME", "NOCUTOFF", "NO_CUTOFF"}:
        raise ValueError("nonbonded_method must be PME or NoCutoff")

    mm, _, _ = import_openmm_modules()
    available_platforms = [mm.Platform.getPlatform(i).getName() for i in range(mm.Platform.getNumPlatforms())]
    if platform not in available_platforms:
        raise ValueError(f"Platform '{platform}' not available. Available: {available_platforms}")

    m_scales = DEFAULT_M_SCALES if m_scales is None else m_scales
    p_scales = DEFAULT_P_SCALES if p_scales is None else p_scales
    d_scales = DEFAULT_D_SCALES if d_scales is None else d_scales
    for name, values in [("m_scales", m_scales), ("p_scales", p_scales), ("d_scales", d_scales)]:
        if len(list(values)) != 5:
            raise ValueError(f"{name} must contain 5 values for [12,13,14,15,16].")

    return {
        "pdb_path": str(pdb),
        "xml_path": str(xml),
        "nonbonded_method": "PME" if method == "PME" else "NoCutoff",
        "platform": platform,
        "m_scales": list(m_scales),
        "p_scales": list(p_scales),
        "d_scales": list(d_scales),
    }


def apply_mpid_scale_exclusions(system, topology, m_scales=None, p_scales=None, d_scales=None):
    """
    Apply MPID covalent maps based on m/p scale exclusion lists.
    Returns True if MPIDForce is found and patched.
    """
    try:
        import mpidplugin  # noqa: F401
    except Exception:
        return False

    m_scales = list(DEFAULT_M_SCALES if m_scales is None else m_scales)
    p_scales = list(DEFAULT_P_SCALES if p_scales is None else p_scales)
    d_scales = list(DEFAULT_D_SCALES if d_scales is None else d_scales)
    if not (len(m_scales) == len(p_scales) == len(d_scales) == 5):
        raise ValueError("m/p/d scales must each have 5 values.")

    mpid_force = None
    for i in range(system.getNumForces()):
        f = system.getForce(i)
        if mpidplugin.MPIDForce.isinstance(f):
            mpid_force = mpidplugin.MPIDForce.cast(f)
            break
    if mpid_force is None:
        return False

    shells_all = _bonded_shells(topology, max_depth=5)
    residue_atoms = [[atom.index for atom in residue.atoms()] for residue in topology.residues()]
    atom_to_residue = {a: r for r, atoms in enumerate(residue_atoms) for a in atoms}
    covalent15 = getattr(mpid_force, "Covalent15", 3)

    for atom_index in range(mpid_force.getNumMultipoles()):
        shells = shells_all[atom_index]
        residue_index = atom_to_residue[atom_index]
        full_intra = tuple(sorted([a for a in residue_atoms[residue_index] if a != atom_index]))

        if float(m_scales[4]) == 0.0:
            c12, c13, c14, c15 = full_intra, tuple(), tuple(), tuple()
        else:
            c12 = tuple(sorted(shells[1])) if float(m_scales[0]) == 0.0 else tuple()
            c13 = tuple(sorted(shells[2])) if float(m_scales[1]) == 0.0 else tuple()
            c14 = tuple(sorted(shells[3])) if float(m_scales[2]) == 0.0 else tuple()
            c15 = tuple(sorted(shells[4])) if float(m_scales[3]) == 0.0 else tuple()
        mpid_force.setCovalentMap(atom_index, mpid_force.Covalent12, c12)
        mpid_force.setCovalentMap(atom_index, mpid_force.Covalent13, c13)
        mpid_force.setCovalentMap(atom_index, mpid_force.Covalent14, c14)
        mpid_force.setCovalentMap(atom_index, covalent15, c15)

        if float(p_scales[4]) == 0.0:
            p12, p13, p14 = full_intra, tuple(), tuple()
        else:
            p12 = tuple(sorted(shells[1])) if float(p_scales[0]) == 0.0 else tuple()
            p13 = tuple(sorted(shells[2])) if float(p_scales[1]) == 0.0 else tuple()
            p14 = tuple(sorted(shells[3])) if float(p_scales[2]) == 0.0 else tuple()
        mpid_force.setCovalentMap(atom_index, mpid_force.PolarizationCovalent11, (atom_index,))
        mpid_force.setCovalentMap(atom_index, mpid_force.PolarizationCovalent12, p12)
        mpid_force.setCovalentMap(atom_index, mpid_force.PolarizationCovalent13, p13)
        mpid_force.setCovalentMap(atom_index, mpid_force.PolarizationCovalent14, p14)

    return True


def load_phyneo_system(
    pdb_path,
    xml_path,
    nonbonded_method="NoCutoff",
    cutoff_nm=1.0,
    constraints="HBonds",
    remove_cmmotion=False,
    use_mpid_scale_exclusions=True,
    m_scales=None,
    p_scales=None,
    d_scales=None,
    set_force_groups=False,
    platform="CPU",
    verbose=True,
):
    """
    Load PDB+XML into OpenMM System with PhyNEO MPID scale handling.
    """
    _vprint(verbose, "validate_inputs: start")
    checked = validate_inputs(
        pdb_path=pdb_path,
        xml_path=xml_path,
        nonbonded_method=nonbonded_method,
        platform=platform,
        m_scales=m_scales,
        p_scales=p_scales,
        d_scales=d_scales,
    )
    _vprint(
        verbose,
        f"validate_inputs: ok | pdb={checked['pdb_path']} | xml={checked['xml_path']} | "
        f"method={checked['nonbonded_method']} | platform={checked['platform']}",
    )

    mm, app, unit = import_openmm_modules()
    plugin_dir = _load_plugin_dir(mm)
    _vprint(verbose, f"openmm import: ok | plugin_dir={plugin_dir if plugin_dir else 'None'}")
    try:
        import mpidplugin  # noqa: F401
        _vprint(verbose, "mpidplugin import: ok")
    except Exception:
        _vprint(verbose, "mpidplugin import: not available")

    _vprint(verbose, "load PDB + XML: start")
    pdb = app.PDBFile(checked["pdb_path"])
    ff = app.ForceField(checked["xml_path"])
    _vprint(verbose, f"load PDB + XML: ok | atoms={pdb.topology.getNumAtoms()}")

    method = app.PME if checked["nonbonded_method"] == "PME" else app.NoCutoff
    kwargs = dict(
        nonbondedMethod=method,
        constraints=_resolve_constraints(app, constraints),
        removeCMMotion=bool(remove_cmmotion),
    )
    if method == app.PME:
        kwargs["nonbondedCutoff"] = float(cutoff_nm) * unit.nanometer

    _vprint(verbose, "createSystem: start")
    system = ff.createSystem(pdb.topology, **kwargs)
    _vprint(verbose, f"createSystem: ok | particles={system.getNumParticles()} | forces={system.getNumForces()}")

    if use_mpid_scale_exclusions:
        mpid_scales_applied = apply_mpid_scale_exclusions(
            system,
            pdb.topology,
            checked["m_scales"],
            checked["p_scales"],
            checked["d_scales"],
        )
        _vprint(
            verbose,
            "apply_mpid_scale_exclusions: "
            f"{'applied' if mpid_scales_applied else 'skipped (no MPIDForce found)'} | "
            f"m={checked['m_scales']} p={checked['p_scales']} d={checked['d_scales']}",
        )
    else:
        mpid_scales_applied = False
        _vprint(verbose, "apply_mpid_scale_exclusions: disabled")

    for force in system.getForces():
        if isinstance(force, mm.CustomNonbondedForce) and method == app.PME:
            force.setNonbondedMethod(mm.CustomNonbondedForce.CutoffPeriodic)
            force.setCutoffDistance(float(cutoff_nm) * unit.nanometer)

    if set_force_groups:
        for i in range(system.getNumForces()):
            system.getForce(i).setForceGroup(i)

    force_names = []
    for i in range(system.getNumForces()):
        f = system.getForce(i)
        name = f.getName() if hasattr(f, "getName") else type(f).__name__
        force_names.append(name)
    _vprint(verbose, f"system forces: {force_names}")
    _vprint(verbose, "load_phyneo_system: done")

    return {
        "mm": mm,
        "app": app,
        "unit": unit,
        "pdb": pdb,
        "system": system,
        "mpid_scales_applied": bool(mpid_scales_applied),
        "validated": checked,
    }


def decompose_openmm_energy(context, system, unit):
    """Return per-force energy decomposition (kJ/mol)."""
    energies = {}
    for i in range(system.getNumForces()):
        e = context.getState(getEnergy=True, groups=1 << i).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        f = system.getForce(i)
        name = f.getName() if hasattr(f, "getName") else type(f).__name__
        key = name if name not in energies else f"{i}_{name}"
        energies[key] = float(e)
    return energies


def _read_csv_column(csv_path: Path, column: str):
    vals = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row is None:
                continue
            v = row.get(column)
            if v in (None, ""):
                continue
            vals.append(float(v))
    return vals


def _mean_std(values):
    if not values:
        return None, None
    n = len(values)
    mean = sum(values) / n
    var = sum((x - mean) ** 2 for x in values) / n
    return mean, var**0.5


@dataclass
class StageConfig:
    name: str
    steps: int
    minimize: bool = False
    traj_interval: int = 500
    state_interval: int = 500
    pressure_atm: float | None = None
    temperature_k: float = 300.0
    timestep_fs: float = 1.0
    friction_ps: float = 1.0


def _run_openmm_stage(
    *,
    loaded,
    out_dir: Path,
    stage: StageConfig,
    platform_name: str,
    positions,
    box_vectors,
    vv_noneq=False,
):
    mm = loaded["mm"]
    app = loaded["app"]
    unit = loaded["unit"]
    pdb = loaded["pdb"]
    system = loaded["system"]

    work_pdb = out_dir / f"{stage.name}_final.pdb"
    dcd_path = out_dir / f"{stage.name}.dcd"
    csv_path = out_dir / f"{stage.name}_state.csv"

    _vprint(True, f"stage[{stage.name}] start | steps={stage.steps} | T={stage.temperature_k} K")
    stage_system = mm.XmlSerializer.deserialize(mm.XmlSerializer.serialize(system))
    use_periodic = bool(stage_system.usesPeriodicBoundaryConditions())
    if stage.pressure_atm is not None and box_vectors is not None and use_periodic:
        stage_system.addForce(
            mm.MonteCarloBarostat(
                float(stage.pressure_atm) * unit.atmosphere,
                float(stage.temperature_k) * unit.kelvin,
            )
        )

    if vv_noneq:
        from velocityverletplugin import VVIntegrator
        from phyneo_openmm.md_utils.viscosity import ViscosityReporter

        integrator = VVIntegrator(
            temperature=float(stage.temperature_k) * unit.kelvin,
            frequency=1.0 / unit.picoseconds,
            drudeTemperature=float(stage.temperature_k) * unit.kelvin,
            drudeFrequency=100 / unit.picoseconds,
            stepSize=float(stage.timestep_fs) * unit.femtoseconds,
            numNHChains=3,
            loopsPerStep=3,
        )
        integrator.setUseMiddleScheme(True)
        integrator.setCosAcceleration(0.02)
    else:
        integrator = mm.LangevinMiddleIntegrator(
            float(stage.temperature_k) * unit.kelvin,
            float(stage.friction_ps) / unit.picosecond,
            float(stage.timestep_fs) * unit.femtoseconds,
        )

    platform_obj = mm.Platform.getPlatformByName(platform_name)
    sim = app.Simulation(pdb.topology, stage_system, integrator, platform_obj)
    sim.context.setPositions(positions)
    if box_vectors is not None and use_periodic:
        sim.context.setPeriodicBoxVectors(*box_vectors)
    if stage.minimize:
        sim.minimizeEnergy(maxIterations=1000)
    sim.context.setVelocitiesToTemperature(float(stage.temperature_k) * unit.kelvin)

    if int(stage.traj_interval) > 0:
        sim.reporters.append(app.DCDReporter(str(dcd_path), int(stage.traj_interval)))
    if vv_noneq:
        vis_path = out_dir / "viscosity.csv"
        sim.reporters.append(ViscosityReporter(str(vis_path), 50))
    elif int(stage.state_interval) > 0:
        sim.reporters.append(
            app.StateDataReporter(
                str(csv_path),
                int(stage.state_interval),
                step=True,
                potentialEnergy=True,
                kineticEnergy=True,
                totalEnergy=True,
                temperature=True,
                volume=use_periodic,
                density=use_periodic,
            )
        )

    sim.step(int(stage.steps))
    state = sim.context.getState(getPositions=True, enforcePeriodicBox=use_periodic)
    with open(work_pdb, "w") as f:
        app.PDBFile.writeFile(sim.topology, state.getPositions(), f)
    out = {
        "stage": stage.name,
        "final_pdb": str(work_pdb),
        "traj": str(dcd_path) if int(stage.traj_interval) > 0 else None,
        "state_csv": str(csv_path) if (not vv_noneq and int(stage.state_interval) > 0) else None,
        "steps": int(stage.steps),
    }
    if vv_noneq:
        out["viscosity_csv"] = str(out_dir / "viscosity.csv")
    _vprint(True, f"stage[{stage.name}] done | final_pdb={work_pdb}")
    return out, state.getPositions(), state.getPeriodicBoxVectors()


class Protocol:  # noqa: D401
    """Packmol-PDB + XML based protocol base class."""

    def __init__(self, config: dict):
        self.config = copy.deepcopy(config)
        self.input = self.config["input"]
        self.runtime = self.config.get("runtime", {})
        self.output = self.config.get("output", {})
        self.out_dir = Path(self.output.get("work_dir", "phyneo_protocol_out")).resolve()
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.loaded = load_phyneo_system(
            pdb_path=self.input["pdb"],
            xml_path=self.input["xml"],
            nonbonded_method=self.runtime.get("nonbonded_method", "PME"),
            cutoff_nm=float(self.runtime.get("cutoff_nm", 1.0)),
            constraints=self.runtime.get("constraints", "HBonds"),
            use_mpid_scale_exclusions=bool(self.runtime.get("mpid_scale_exclusions", True)),
            m_scales=self.runtime.get("m_scales", DEFAULT_M_SCALES),
            p_scales=self.runtime.get("p_scales", DEFAULT_P_SCALES),
            d_scales=self.runtime.get("d_scales", DEFAULT_D_SCALES),
            platform=self.runtime.get("platform", "CPU"),
        )
        self.platform = self.runtime.get("platform", "CPU")
        self.outputs = []
        self.positions = self.loaded["pdb"].positions
        self.box_vectors = self.loaded["pdb"].topology.getPeriodicBoxVectors()

    def run_stage(self, stage: StageConfig, vv_noneq=False):
        out, pos, box = _run_openmm_stage(
            loaded=self.loaded,
            out_dir=self.out_dir,
            stage=stage,
            platform_name=self.platform,
            positions=self.positions,
            box_vectors=self.box_vectors,
            vv_noneq=vv_noneq,
        )
        self.positions = pos
        self.box_vectors = box
        self.outputs.append(out)
        return out

    def run_protocol(self):
        raise NotImplementedError

    def post_process(self):
        return {}


class DensityProtocol(Protocol):
    def run_protocol(self):
        cfg = self.config.get("stages", {}).get("npt", {})
        stage = StageConfig(
            name="npt",
            steps=int(cfg.get("steps", 200000)),
            minimize=bool(cfg.get("minimize", True)),
            traj_interval=int(cfg.get("traj_interval", 500)),
            state_interval=int(cfg.get("state_interval", 500)),
            pressure_atm=float(cfg.get("pressure_atm", 1.0)),
            temperature_k=float(cfg.get("temperature_k", self.runtime.get("temperature_k", 300.0))),
            timestep_fs=float(cfg.get("timestep_fs", self.runtime.get("timestep_fs", 1.0))),
            friction_ps=float(cfg.get("friction_ps", self.runtime.get("friction_ps", 1.0))),
        )
        self.run_stage(stage)
        return {"outputs": self.outputs}

    def post_process(self):
        npt_csv = self.out_dir / "npt_state.csv"
        if not npt_csv.exists():
            return {}
        density = _read_csv_column(npt_csv, "Density (g/mL)")
        if not density:
            return {}
        tail = density[max(0, int(0.5 * len(density))) :]
        mean, std = _mean_std(tail)
        return {"density": float(mean), "density_std": float(std)}


class TransportProtocol(Protocol):
    def run_protocol(self):
        stages_cfg = self.config.get("stages", {})
        npt = stages_cfg.get("npt", {})
        nvt = stages_cfg.get("nvt", {})
        noneq = stages_cfg.get("noneq", {})
        self.run_stage(
            StageConfig(
                name="npt",
                steps=int(npt.get("steps", 400000)),
                minimize=bool(npt.get("minimize", True)),
                traj_interval=int(npt.get("traj_interval", 500)),
                state_interval=int(npt.get("state_interval", 500)),
                pressure_atm=float(npt.get("pressure_atm", 1.0)),
                temperature_k=float(npt.get("temperature_k", self.runtime.get("temperature_k", 300.0))),
                timestep_fs=float(npt.get("timestep_fs", self.runtime.get("timestep_fs", 1.0))),
                friction_ps=float(npt.get("friction_ps", self.runtime.get("friction_ps", 1.0))),
            )
        )
        self.run_stage(
            StageConfig(
                name="nvt",
                steps=int(nvt.get("steps", 1000000)),
                minimize=bool(nvt.get("minimize", False)),
                traj_interval=int(nvt.get("traj_interval", 500)),
                state_interval=int(nvt.get("state_interval", 500)),
                pressure_atm=None,
                temperature_k=float(nvt.get("temperature_k", self.runtime.get("temperature_k", 300.0))),
                timestep_fs=float(nvt.get("timestep_fs", self.runtime.get("timestep_fs", 1.0))),
                friction_ps=float(nvt.get("friction_ps", self.runtime.get("friction_ps", 1.0))),
            )
        )
        if bool(noneq.get("enabled", True)):
            self.run_stage(
                StageConfig(
                    name="noneq",
                    steps=int(noneq.get("steps", 100000)),
                    minimize=False,
                    traj_interval=int(noneq.get("traj_interval", 0)),
                    state_interval=0,
                    pressure_atm=None,
                    temperature_k=float(noneq.get("temperature_k", self.runtime.get("temperature_k", 300.0))),
                    timestep_fs=float(noneq.get("timestep_fs", 1.0)),
                    friction_ps=float(noneq.get("friction_ps", self.runtime.get("friction_ps", 1.0))),
                ),
                vv_noneq=True,
            )
        return {"outputs": self.outputs}

    def post_process(self):
        out = {}
        try:
            from phyneo_openmm.md_utils.viscosity import viscosity_calc

            out["viscosity_cP"] = float(viscosity_calc(str(self.out_dir)))
        except Exception:
            pass
        try:
            from phyneo_openmm.md_utils.md_run import dcd_read, volume_calc
            from phyneo_openmm.md_utils.onsager_conductivity import onsager_calc

            species_mass = self.config.get("species_mass")
            species_number = self.config.get("species_number")
            species_charge = self.config.get("species_charge")
            if species_mass and species_number and species_charge:
                positions = dcd_read(str(self.out_dir / "nvt.dcd"))
                volume_ang3, temp_k = volume_calc(str(self.out_dir))
                vis = float(out.get("viscosity_cP", self.config.get("viscosity_cP", 1.0)))
                out["onsager"] = onsager_calc(
                    species_mass,
                    species_number,
                    species_charge,
                    volume_ang3,
                    vis,
                    temp_k,
                    positions,
                )
        except Exception:
            pass
        return out


class HVapProtocol(Protocol):
    def run_protocol(self):
        stages_cfg = self.config.get("stages", {})
        npt = stages_cfg.get("npt", {})
        gas_nvt = stages_cfg.get("gas_nvt", {})
        self.run_stage(
            StageConfig(
                name="npt",
                steps=int(npt.get("steps", 300000)),
                minimize=bool(npt.get("minimize", True)),
                traj_interval=int(npt.get("traj_interval", 500)),
                state_interval=int(npt.get("state_interval", 500)),
                pressure_atm=float(npt.get("pressure_atm", 1.0)),
                temperature_k=float(npt.get("temperature_k", self.runtime.get("temperature_k", 300.0))),
                timestep_fs=float(npt.get("timestep_fs", self.runtime.get("timestep_fs", 1.0))),
                friction_ps=float(npt.get("friction_ps", self.runtime.get("friction_ps", 1.0))),
            )
        )
        # Gas phase: optional external PDB with single molecule in big box.
        gas_pdb = self.config.get("input", {}).get("gas_pdb")
        if gas_pdb:
            gas_loaded = load_phyneo_system(
                pdb_path=gas_pdb,
                xml_path=self.input["xml"],
                nonbonded_method="NoCutoff",
                constraints=self.runtime.get("constraints", "HBonds"),
                use_mpid_scale_exclusions=bool(self.runtime.get("mpid_scale_exclusions", True)),
                m_scales=self.runtime.get("m_scales", DEFAULT_M_SCALES),
                p_scales=self.runtime.get("p_scales", DEFAULT_P_SCALES),
                d_scales=self.runtime.get("d_scales", DEFAULT_D_SCALES),
                platform=self.runtime.get("platform", "CPU"),
            )
            orig_loaded = self.loaded
            orig_pos, orig_box = self.positions, self.box_vectors
            self.loaded = gas_loaded
            self.positions = gas_loaded["pdb"].positions
            self.box_vectors = gas_loaded["pdb"].topology.getPeriodicBoxVectors()
            self.run_stage(
                StageConfig(
                    name="gas_nvt",
                    steps=int(gas_nvt.get("steps", 500000)),
                    minimize=bool(gas_nvt.get("minimize", False)),
                    traj_interval=int(gas_nvt.get("traj_interval", 500)),
                    state_interval=int(gas_nvt.get("state_interval", 500)),
                    pressure_atm=None,
                    temperature_k=float(gas_nvt.get("temperature_k", self.runtime.get("temperature_k", 300.0))),
                    timestep_fs=float(gas_nvt.get("timestep_fs", 0.2)),
                    friction_ps=float(gas_nvt.get("friction_ps", self.runtime.get("friction_ps", 1.0))),
                )
            )
            self.loaded = orig_loaded
            self.positions, self.box_vectors = orig_pos, orig_box
        return {"outputs": self.outputs}

    def post_process(self):
        out = {}
        npt_csv = self.out_dir / "npt_state.csv"
        if npt_csv.exists():
            d = _read_csv_column(npt_csv, "Density (g/mL)")
            if d:
                d_tail = d[max(0, int(0.5 * len(d))) :]
                d_mean, _ = _mean_std(d_tail)
                out["density"] = float(d_mean)
            e_liq = _read_csv_column(npt_csv, "Potential Energy (kJ/mole)")
            if e_liq:
                e_tail = e_liq[max(0, int(0.5 * len(e_liq))) :]
                e_mean, _ = _mean_std(e_tail)
                out["e_liquid_kj_mol"] = float(e_mean)
        gas_csv = self.out_dir / "gas_nvt_state.csv"
        if gas_csv.exists():
            e_gas_list = _read_csv_column(gas_csv, "Potential Energy (kJ/mole)")
            if e_gas_list and "e_liquid_kj_mol" in out:
                e_gas_mean, _ = _mean_std(e_gas_list)
                e_gas = float(e_gas_mean)
                out["e_gas_kj_mol"] = e_gas
                t = float(self.runtime.get("temperature_k", 300.0))
                out["hvap_kcal_mol"] = float((e_gas - out["e_liquid_kj_mol"]) / 4.184 + 8.314 * t / 1000.0 / 4.184)
        return out


def _run_generic_stages(config):
    """Minimal staged runner (npt/nvt/md) for PhyNEO XML+PDB workflows."""
    if isinstance(config, (str, Path)):
        cfg = json.loads(Path(config).read_text())
    else:
        cfg = dict(config)

    input_cfg = cfg["input"]
    runtime = cfg.get("runtime", {})
    out_cfg = cfg.get("output", {})
    stages = cfg.get("stages", [])
    out_dir = Path(out_cfg.get("work_dir", "phyneo_protocol_out")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    loaded = load_phyneo_system(
        pdb_path=input_cfg["pdb"],
        xml_path=input_cfg["xml"],
        nonbonded_method=runtime.get("nonbonded_method", "PME"),
        cutoff_nm=float(runtime.get("cutoff_nm", 1.0)),
        constraints=runtime.get("constraints", "HBonds"),
        remove_cmmotion=False,
        use_mpid_scale_exclusions=bool(runtime.get("mpid_scale_exclusions", True)),
        m_scales=runtime.get("m_scales", DEFAULT_M_SCALES),
        p_scales=runtime.get("p_scales", DEFAULT_P_SCALES),
        d_scales=runtime.get("d_scales", DEFAULT_D_SCALES),
        set_force_groups=False,
        platform=runtime.get("platform", "CPU"),
    )

    mm = loaded["mm"]
    app = loaded["app"]
    unit = loaded["unit"]
    pdb = loaded["pdb"]
    system = loaded["system"]
    platform_obj = mm.Platform.getPlatformByName(runtime.get("platform", "CPU"))

    positions = pdb.positions
    if pdb.topology.getPeriodicBoxVectors() is not None:
        box_vectors = pdb.topology.getPeriodicBoxVectors()
    else:
        box_vectors = None

    outputs = []
    for raw_stage in stages:
        stage = StageConfig(**raw_stage)
        work_pdb = out_dir / f"{stage.name}_final.pdb"
        dcd_path = out_dir / f"{stage.name}.dcd"
        csv_path = out_dir / f"{stage.name}_state.csv"

        stage_system = mm.XmlSerializer.deserialize(mm.XmlSerializer.serialize(system))
        if stage.pressure_atm is not None and box_vectors is not None:
            stage_system.addForce(
                mm.MonteCarloBarostat(
                    float(stage.pressure_atm) * unit.atmosphere,
                    float(stage.temperature_k) * unit.kelvin,
                )
            )

        integrator = mm.LangevinMiddleIntegrator(
            float(stage.temperature_k) * unit.kelvin,
            float(stage.friction_ps) / unit.picosecond,
            float(stage.timestep_fs) * unit.femtoseconds,
        )
        sim = app.Simulation(pdb.topology, stage_system, integrator, platform_obj)
        sim.context.setPositions(positions)
        if box_vectors is not None:
            sim.context.setPeriodicBoxVectors(*box_vectors)
        if stage.minimize:
            sim.minimizeEnergy(maxIterations=1000)
        sim.context.setVelocitiesToTemperature(float(stage.temperature_k) * unit.kelvin)
        if int(stage.traj_interval) > 0:
            sim.reporters.append(app.DCDReporter(str(dcd_path), int(stage.traj_interval)))
        if int(stage.state_interval) > 0:
            sim.reporters.append(
                app.StateDataReporter(
                    str(csv_path),
                    int(stage.state_interval),
                    step=True,
                    potentialEnergy=True,
                    kineticEnergy=True,
                    totalEnergy=True,
                    temperature=True,
                    volume=box_vectors is not None,
                    density=box_vectors is not None,
                )
            )

        sim.step(int(stage.steps))
        state = sim.context.getState(getPositions=True, enforcePeriodicBox=(box_vectors is not None))
        with open(work_pdb, "w") as f:
            app.PDBFile.writeFile(sim.topology, state.getPositions(), f)
        positions = state.getPositions()
        box_vectors = state.getPeriodicBoxVectors()
        outputs.append(
            {
                "stage": stage.name,
                "final_pdb": str(work_pdb),
                "traj": str(dcd_path) if int(stage.traj_interval) > 0 else None,
                "state_csv": str(csv_path) if int(stage.state_interval) > 0 else None,
                "steps": int(stage.steps),
            }
        )

    result = {
        "validated": loaded["validated"],
        "mpid_scales_applied": loaded["mpid_scales_applied"],
        "outputs": outputs,
    }
    summary = out_dir / str(out_cfg.get("summary_json", "phyneo_protocol_summary.json"))
    summary.write_text(json.dumps(result, indent=2))
    return result


def run_protocol(config):
    """
    Run protocol by type.
    Supported types:
    - density
    - transport
    - hvap
    - generic (fallback to explicit stages list)
    """
    if isinstance(config, (str, Path)):
        cfg = json.loads(Path(config).read_text())
    else:
        cfg = dict(config)

    ptype = str(cfg.get("protocol", {}).get("type", cfg.get("type", "generic"))).lower()
    if ptype == "density":
        runner = DensityProtocol(cfg)
    elif ptype == "transport":
        runner = TransportProtocol(cfg)
    elif ptype == "hvap":
        runner = HVapProtocol(cfg)
    else:
        return _run_generic_stages(cfg)

    run_out = runner.run_protocol()
    post = runner.post_process()
    summary = {
        "validated": runner.loaded["validated"],
        "mpid_scales_applied": runner.loaded["mpid_scales_applied"],
        "outputs": run_out.get("outputs", []),
        "post_process": post,
    }
    summary_path = runner.out_dir / str(runner.output.get("summary_json", "phyneo_protocol_summary.json"))
    summary_path.write_text(json.dumps(summary, indent=2))
    return summary


def _main():
    parser = argparse.ArgumentParser(description="Minimal PhyNEO protocol")
    parser.add_argument("--config", required=True, help="JSON config path for run_protocol")
    args = parser.parse_args()
    out = run_protocol(args.config)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    _main()


__all__ = [
    # New minimal validated loading path.
    "DEFAULT_M_SCALES",
    "DEFAULT_P_SCALES",
    "DEFAULT_D_SCALES",
    "validate_inputs",
    "apply_mpid_scale_exclusions",
    "load_phyneo_system",
    "decompose_openmm_energy",
    "run_protocol",
    # Full legacy protocol API.
    "ComponentType",
    "Component",
    "predict_density",
    "search_mixture",
    "predict_box",
    "load_topo",
    "generate_system_gro",
    "write_gro",
    "Protocol",
    "DensityProtocol",
    "TransportProtocol",
    "HVapProtocol",
]
