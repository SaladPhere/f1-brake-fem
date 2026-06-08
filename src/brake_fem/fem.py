from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .mesh import AnnulusMesh


@dataclass(frozen=True)
class FEMOperators:
    mass: np.ndarray
    node_area: np.ndarray
    stiffness_diag: np.ndarray
    stiffness_i: np.ndarray
    stiffness_j: np.ndarray
    stiffness_v: np.ndarray
    cooling_weights: np.ndarray
    total_area: float
    thermal_capacity_total: float


def assemble_heat_operators(mesh: AnnulusMesh, config: dict) -> FEMOperators:
    geom = config["geometry"]
    mat = config["material"]
    cooling = config.get("cooling", {})

    thickness_eff = float(geom["thickness"]) * float(geom.get("solid_fraction", 1.0))
    rho = float(mat["rho"])
    cp = float(mat["cp"])
    k = float(mat.get("k_inplane", 0.0))
    cooling_area_factor = float(cooling.get("cooling_area_factor", 2.0))

    n_nodes = mesh.node_count
    mass = np.zeros(n_nodes, dtype=float)
    node_area = np.zeros(n_nodes, dtype=float)
    k_diag = np.zeros(n_nodes, dtype=float)
    offdiag: dict[tuple[int, int], float] = {}

    for tri in mesh.elements:
        coords = mesh.nodes[tri]
        area, grads = _triangle_geometry(coords)
        if area <= 0.0:
            raise ValueError("Mesh contains a non-positive triangle.")

        area_lump = area / 3.0
        for local, node in enumerate(tri):
            node_area[node] += area_lump
            mass[node] += rho * cp * thickness_eff * area_lump

        local_k = k * thickness_eff * area * (grads @ grads.T)
        for a in range(3):
            ia = int(tri[a])
            k_diag[ia] += local_k[a, a]
            for b in range(a + 1, 3):
                ib = int(tri[b])
                key = (ia, ib) if ia < ib else (ib, ia)
                offdiag[key] = offdiag.get(key, 0.0) + local_k[a, b]

    cooling_weights = cooling_area_factor * node_area.copy()
    if bool(cooling.get("include_edge_cooling", True)):
        edge_weight = float(geom["thickness"]) * float(geom.get("solid_fraction", 1.0))
        for edges in (mesh.inner_boundary_edges, mesh.outer_boundary_edges):
            for a, b in edges:
                length = float(np.linalg.norm(mesh.nodes[a] - mesh.nodes[b]))
                side_area = edge_weight * length
                cooling_weights[a] += 0.5 * side_area
                cooling_weights[b] += 0.5 * side_area

    if offdiag:
        pairs = np.asarray(list(offdiag.keys()), dtype=np.int32)
        values = np.asarray(list(offdiag.values()), dtype=float)
        ii = pairs[:, 0]
        jj = pairs[:, 1]
    else:
        ii = np.asarray([], dtype=np.int32)
        jj = np.asarray([], dtype=np.int32)
        values = np.asarray([], dtype=float)

    return FEMOperators(
        mass=mass,
        node_area=node_area,
        stiffness_diag=k_diag,
        stiffness_i=ii,
        stiffness_j=jj,
        stiffness_v=values,
        cooling_weights=cooling_weights,
        total_area=float(np.sum(node_area)),
        thermal_capacity_total=float(np.sum(mass)),
    )


def apply_stiffness(ops: FEMOperators, x: np.ndarray) -> np.ndarray:
    y = ops.stiffness_diag * x
    if len(ops.stiffness_v):
        np.add.at(y, ops.stiffness_i, ops.stiffness_v * x[ops.stiffness_j])
        np.add.at(y, ops.stiffness_j, ops.stiffness_v * x[ops.stiffness_i])
    return y


def operator_matvec(
    diagonal: np.ndarray,
    i_idx: np.ndarray,
    j_idx: np.ndarray,
    values: np.ndarray,
    x: np.ndarray,
) -> np.ndarray:
    y = diagonal * x
    if len(values):
        np.add.at(y, i_idx, values * x[j_idx])
        np.add.at(y, j_idx, values * x[i_idx])
    return y


def _triangle_geometry(coords: np.ndarray) -> tuple[float, np.ndarray]:
    x0, y0 = coords[0]
    x1, y1 = coords[1]
    x2, y2 = coords[2]
    area2 = (x1 - x0) * (y2 - y0) - (y1 - y0) * (x2 - x0)
    area = 0.5 * area2
    if area <= 0.0:
        return area, np.zeros((3, 2), dtype=float)
    b = np.array([y1 - y2, y2 - y0, y0 - y1], dtype=float)
    c = np.array([x2 - x1, x0 - x2, x1 - x0], dtype=float)
    grads = np.column_stack((b, c)) / (2.0 * area)
    return area, grads
