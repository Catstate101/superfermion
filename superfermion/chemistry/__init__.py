"""
Quantum Chemistry — Molecular simulation and electronic structure.

Heavy dependencies (pyscf, scipy) are loaded lazily on first access.
"""

__all__ = [
    "FermionicOperator", "get_molecular_hamiltonian", "uccsd_ansatz",
    "molecule_from_geometry", "molecule_from_xyz",
    "MolecularData", "active_space_from_homo_lumo",
    # active-space selection
    "taper_z2", "z2_symmetries",
    "orbital_entropy", "single_orbital_entropy", "mutual_information",
    "autocas_selection", "entropies_from_integrals",
    "avas_active_space", "aegiss_selection", "optimise_orbital_basis",
    # solvers and embeddings
    "exact_energy", "all_determinants", "hamiltonian_matrix",
    "sqd_energy", "qsci", "one_shot_dmet",
    # encodings and file formats
    "scbk_from_integrals",
    "read_fcidump", "write_fcidump", "read_fcidump_text",
    "write_fcidump_text", "to_hdf5", "from_hdf5",
]

_LAZY_MAP = {
    "FermionicOperator":         "superfermion.chemistry.hamiltonians",
    "get_molecular_hamiltonian": "superfermion.chemistry.hamiltonians",
    "uccsd_ansatz":              "superfermion.chemistry.ansatz",
    "molecule_from_geometry":    "superfermion.chemistry.pyscf_bridge",
    "molecule_from_xyz":         "superfermion.chemistry.pyscf_bridge",
    "MolecularData":             "superfermion.chemistry.pyscf_bridge",
    "active_space_from_homo_lumo": "superfermion.chemistry.pyscf_bridge",
    "taper_z2":                  "superfermion.chemistry.tapering",
    "z2_symmetries":             "superfermion.chemistry.tapering",
    "orbital_entropy":           "superfermion.chemistry.entropy",
    "single_orbital_entropy":    "superfermion.chemistry.entropy",
    "mutual_information":        "superfermion.chemistry.entropy",
    "autocas_selection":         "superfermion.chemistry.entropy",
    "entropies_from_integrals":  "superfermion.chemistry.entropy",
    "avas_active_space":         "superfermion.chemistry.avas",
    "aegiss_selection":          "superfermion.chemistry.aegiss",
    "optimise_orbital_basis":    "superfermion.chemistry.orbital_reduction",
    "exact_energy":              "superfermion.chemistry.subspace",
    "all_determinants":          "superfermion.chemistry.subspace",
    "hamiltonian_matrix":        "superfermion.chemistry.subspace",
    "sqd_energy":                "superfermion.chemistry.sqd",
    "qsci":                      "superfermion.chemistry.sqd",
    "one_shot_dmet":             "superfermion.chemistry.dmet",
    "scbk_from_integrals":       "superfermion.chemistry.scbk",
    "read_fcidump":              "superfermion.chemistry.fcidump",
    "write_fcidump":             "superfermion.chemistry.fcidump",
    "read_fcidump_text":         "superfermion.chemistry.fcidump",
    "write_fcidump_text":        "superfermion.chemistry.fcidump",
    "to_hdf5":                   "superfermion.chemistry.fcidump",
    "from_hdf5":                 "superfermion.chemistry.fcidump",
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
