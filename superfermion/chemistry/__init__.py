"""
Quantum Chemistry — Molecular simulation and electronic structure.

Heavy dependencies (pyscf, scipy) are loaded lazily on first access.
"""

__all__ = [
    "FermionicOperator", "get_molecular_hamiltonian", "uccsd_ansatz",
    "molecule_from_geometry", "molecule_from_xyz",
    "MolecularData", "active_space_from_homo_lumo",
]

_LAZY_MAP = {
    "FermionicOperator":         "superfermion.chemistry.hamiltonians",
    "get_molecular_hamiltonian": "superfermion.chemistry.hamiltonians",
    "uccsd_ansatz":              "superfermion.chemistry.ansatz",
    "molecule_from_geometry":    "superfermion.chemistry.pyscf_bridge",
    "molecule_from_xyz":         "superfermion.chemistry.pyscf_bridge",
    "MolecularData":             "superfermion.chemistry.pyscf_bridge",
    "active_space_from_homo_lumo": "superfermion.chemistry.pyscf_bridge",
}


def __getattr__(name):
    if name == "_LAZY_MAP":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    mod_name = _LAZY_MAP.get(name)
    if mod_name is not None:
        import importlib
        mod = importlib.import_module(mod_name)
        attr = getattr(mod, name)
        globals()[name] = attr
        return attr
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
