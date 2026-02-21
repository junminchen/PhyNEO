"""phyneo_openmm package.

Use lazy exports to avoid importing `phyneo_protocol` during package import.
This prevents runpy warning when running `python -m phyneo_openmm.phyneo_protocol`.
"""

from importlib import import_module

__all__ = [
    "DEFAULT_M_SCALES",
    "DEFAULT_P_SCALES",
    "DEFAULT_D_SCALES",
    "validate_inputs",
    "apply_mpid_scale_exclusions",
    "load_phyneo_system",
    "run_protocol",
]


def __getattr__(name):
    if name in __all__:
        mod = import_module("phyneo_openmm.phyneo_protocol")
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
