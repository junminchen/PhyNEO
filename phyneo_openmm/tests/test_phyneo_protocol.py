#!/usr/bin/env python3
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]

import sys

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from phyneo_openmm.phyneo_protocol import (
    DEFAULT_D_SCALES,
    DEFAULT_M_SCALES,
    DEFAULT_P_SCALES,
    apply_mpid_scale_exclusions,
    decompose_openmm_energy,
    import_openmm_modules,
    load_phyneo_system,
    validate_inputs,
)


class TestPhyNEOProtocol(unittest.TestCase):
    def test_validate_inputs(self):
        out = validate_inputs(
            pdb_path=REPO_ROOT / "example/3_load_ff_xml/dimer_bank/dimer_003_EC_EC.pdb",
            xml_path=REPO_ROOT / "example/1_make_mpid_xml_wt_bond/caff_5_mpid_slater_bond.xml",
            nonbonded_method="NoCutoff",
            platform="Reference",
            m_scales=DEFAULT_M_SCALES,
            p_scales=DEFAULT_P_SCALES,
            d_scales=DEFAULT_D_SCALES,
        )
        self.assertEqual(out["nonbonded_method"], "NoCutoff")
        self.assertEqual(len(out["m_scales"]), 5)

    def test_load_system_and_mpid_scales(self):
        loaded = load_phyneo_system(
            pdb_path=REPO_ROOT / "example/3_load_ff_xml/dimer_bank/dimer_003_EC_EC.pdb",
            xml_path=REPO_ROOT / "example/1_make_mpid_xml_wt_bond/caff_5_mpid_slater_bond.xml",
            nonbonded_method="NoCutoff",
            constraints=None,
            use_mpid_scale_exclusions=True,
            set_force_groups=True,
            platform="Reference",
        )
        self.assertTrue(loaded["mpid_scales_applied"])

        mm = loaded["mm"]
        unit = loaded["unit"]
        context = mm.Context(loaded["system"], mm.VerletIntegrator(0.001), mm.Platform.getPlatformByName("Reference"))
        context.setPositions(loaded["pdb"].positions)
        decomp = decompose_openmm_energy(context, loaded["system"], unit)
        self.assertIn("MPIDForce", decomp)
        self.assertIn("CustomNonbondedForce", decomp)

    def test_apply_scale_direct_call(self):
        mm, app, _ = import_openmm_modules()
        try:
            import mpidplugin  # noqa: F401
        except Exception:
            self.skipTest("mpidplugin not available")
            return
        pdb = app.PDBFile(str(REPO_ROOT / "example/3_load_ff_xml/dimer_bank/dimer_003_EC_EC.pdb"))
        ff = app.ForceField(str(REPO_ROOT / "example/1_make_mpid_xml_wt_bond/caff_5_mpid_slater_bond.xml"))
        system = ff.createSystem(pdb.topology, nonbondedMethod=app.NoCutoff, constraints=None, removeCMMotion=False)
        ok = apply_mpid_scale_exclusions(system, pdb.topology, DEFAULT_M_SCALES, DEFAULT_P_SCALES, DEFAULT_D_SCALES)
        self.assertTrue(ok)


if __name__ == "__main__":
    unittest.main()
