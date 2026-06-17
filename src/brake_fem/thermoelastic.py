from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .fem import operator_matvec
from .mesh import AnnulusMesh
from .solver import SimulationResult, pcg_solve


@dataclass(frozen=True)
class ElasticityOperators:
    diagonal: np.ndarray
    i_idx: np.ndarray
    j_idx: np.ndarray
    values: np.ndarray
    node_area: np.ndarray
    element_area: np.ndarray
    element_b: np.ndarray
    elastic_matrix: np.ndarray
    constrained_dofs: np.ndarray
    free_dofs: np.ndarray
    thickness_eff: float
    model: str


@dataclass
class ThermoelasticResult:
    times: np.ndarray
    displacements: np.ndarray
    nodal_von_mises: np.ndarray
    element_von_mises: np.ndarray
    max_von_mises_pa: np.ndarray
    mean_von_mises_pa: np.ndarray
    max_displacement_m: np.ndarray
    solver_iterations: np.ndarray
    solver_residuals: np.ndarray
    reference_temperature_c: float
    config: dict[str, Any]


def solve_thermoelastic_snapshots(result: SimulationResult) -> ThermoelasticResult | None:
    cfg = result.config.get("thermoelastic", {})
    if not bool(cfg.get("enabled", False)) or len(result.snapshots) == 0:
        return None

    ops = assemble_elasticity_operators(result.mesh, result.config)
    reference_c = _reference_temperature(result, cfg)
    tol = float(cfg.get("cg_tol", 1.0e-8))
    max_iter = int(cfg.get("cg_max_iter", 800))

    n_snap = len(result.snapshots)
    n_nodes = result.mesh.node_count
    n_elem = result.mesh.element_count
    displacements = np.zeros((n_snap, n_nodes, 2), dtype=np.float32)
    nodal_vm = np.zeros((n_snap, n_nodes), dtype=np.float32)
    elem_vm = np.zeros((n_snap, n_elem), dtype=np.float32)
    max_vm = np.zeros(n_snap, dtype=float)
    mean_vm = np.zeros(n_snap, dtype=float)
    max_disp = np.zeros(n_snap, dtype=float)
    iterations = np.zeros(n_snap, dtype=int)
    residuals = np.zeros(n_snap, dtype=float)

    x0 = np.zeros(n_nodes * 2, dtype=float)
    for idx, temp in enumerate(result.snapshots.astype(float)):
        if _is_spatially_uniform(temp, cfg):
            u = np.zeros(n_nodes * 2, dtype=float)
            iters = 0
            rel_res = 0.0
        else:
            load = thermal_load_vector(result.mesh, ops, temp, reference_c, result.config)
            u, iters, rel_res = constrained_pcg_solve(ops, load, x0, tol=tol, max_iter=max_iter)
        x0 = u
        stress = recover_von_mises(result.mesh, ops, temp, u, reference_c, result.config)
        disp = u.reshape(n_nodes, 2)
        nodal = element_to_node_values(result.mesh, ops, stress)

        displacements[idx] = disp.astype(np.float32)
        elem_vm[idx] = stress.astype(np.float32)
        nodal_vm[idx] = nodal.astype(np.float32)
        max_vm[idx] = float(np.max(stress))
        mean_vm[idx] = float(np.average(stress, weights=ops.element_area))
        max_disp[idx] = float(np.max(np.linalg.norm(disp, axis=1)))
        iterations[idx] = iters
        residuals[idx] = rel_res

    return ThermoelasticResult(
        times=result.snapshots_t.copy(),
        displacements=displacements,
        nodal_von_mises=nodal_vm,
        element_von_mises=elem_vm,
        max_von_mises_pa=max_vm,
        mean_von_mises_pa=mean_vm,
        max_displacement_m=max_disp,
        solver_iterations=iterations,
        solver_residuals=residuals,
        reference_temperature_c=reference_c,
        config=result.config,
    )


def assemble_elasticity_operators(mesh: AnnulusMesh, config: dict[str, Any]) -> ElasticityOperators:
    cfg = config.get("thermoelastic", {})
    geom = config["geometry"]
    material = cfg.get("material", {})
    young = float(material.get("young_modulus_pa", 30.0e9))
    poisson = float(material.get("poisson_ratio", 0.22))
    model = str(cfg.get("model", "plane_stress"))
    thickness_eff = float(geom["thickness"])
    if bool(cfg.get("use_solid_fraction", True)):
        thickness_eff *= float(geom.get("solid_fraction", 1.0))

    d_mat = elasticity_matrix(young, poisson, model)
    n_dof = mesh.node_count * 2
    diagonal = np.zeros(n_dof, dtype=float)
    offdiag: dict[tuple[int, int], float] = {}
    node_area = np.zeros(mesh.node_count, dtype=float)
    element_area = np.zeros(mesh.element_count, dtype=float)
    element_b = np.zeros((mesh.element_count, 3, 6), dtype=float)

    for elem_id, tri in enumerate(mesh.elements):
        coords = mesh.nodes[tri]
        area, b_mat = triangle_elastic_b_matrix(coords)
        if area <= 0.0:
            raise ValueError("Mesh contains a non-positive triangle.")
        element_area[elem_id] = area
        element_b[elem_id] = b_mat
        for node in tri:
            node_area[node] += area / 3.0

        local_k = thickness_eff * area * (b_mat.T @ d_mat @ b_mat)
        dofs = _element_dofs(tri)
        for a, ia in enumerate(dofs):
            diagonal[ia] += local_k[a, a]
            for b in range(a + 1, 6):
                ib = dofs[b]
                key = (ia, ib) if ia < ib else (ib, ia)
                offdiag[key] = offdiag.get(key, 0.0) + local_k[a, b]

    if offdiag:
        pairs = np.asarray(list(offdiag.keys()), dtype=np.int32)
        values = np.asarray(list(offdiag.values()), dtype=float)
        i_idx = pairs[:, 0]
        j_idx = pairs[:, 1]
    else:
        i_idx = np.asarray([], dtype=np.int32)
        j_idx = np.asarray([], dtype=np.int32)
        values = np.asarray([], dtype=float)

    constrained = rigid_body_constraints(mesh)
    free_mask = np.ones(n_dof, dtype=bool)
    free_mask[constrained] = False
    free = np.flatnonzero(free_mask).astype(np.int32)

    return ElasticityOperators(
        diagonal=diagonal,
        i_idx=i_idx,
        j_idx=j_idx,
        values=values,
        node_area=node_area,
        element_area=element_area,
        element_b=element_b,
        elastic_matrix=d_mat,
        constrained_dofs=constrained,
        free_dofs=free,
        thickness_eff=thickness_eff,
        model=model,
    )


def elasticity_matrix(young_modulus: float, poisson: float, model: str) -> np.ndarray:
    if not (-0.99 < poisson < 0.49):
        raise ValueError("Poisson ratio must be in a stable range.")
    if young_modulus <= 0.0:
        raise ValueError("Young's modulus must be positive.")

    if model == "plane_strain":
        scale = young_modulus / ((1.0 + poisson) * (1.0 - 2.0 * poisson))
        return scale * np.array(
            [
                [1.0 - poisson, poisson, 0.0],
                [poisson, 1.0 - poisson, 0.0],
                [0.0, 0.0, 0.5 * (1.0 - 2.0 * poisson)],
            ],
            dtype=float,
        )
    if model != "plane_stress":
        raise ValueError("Thermoelastic model must be 'plane_stress' or 'plane_strain'.")
    scale = young_modulus / (1.0 - poisson**2)
    return scale * np.array(
        [
            [1.0, poisson, 0.0],
            [poisson, 1.0, 0.0],
            [0.0, 0.0, 0.5 * (1.0 - poisson)],
        ],
        dtype=float,
    )


def thermal_load_vector(
    mesh: AnnulusMesh,
    ops: ElasticityOperators,
    temperature_c: np.ndarray,
    reference_temperature_c: float,
    config: dict[str, Any],
) -> np.ndarray:
    alpha = _thermal_expansion(config)
    load = np.zeros(mesh.node_count * 2, dtype=float)
    thermal_shape = np.array([1.0, 1.0, 0.0], dtype=float)

    for elem_id, tri in enumerate(mesh.elements):
        d_temp = float(np.mean(temperature_c[tri]) - reference_temperature_c)
        eps_th = alpha * d_temp * thermal_shape
        local = ops.thickness_eff * ops.element_area[elem_id] * (ops.element_b[elem_id].T @ ops.elastic_matrix @ eps_th)
        dofs = _element_dofs(tri)
        np.add.at(load, dofs, local)
    load[ops.constrained_dofs] = 0.0
    return load


def constrained_pcg_solve(
    ops: ElasticityOperators,
    rhs: np.ndarray,
    x0: np.ndarray | None,
    tol: float,
    max_iter: int,
) -> tuple[np.ndarray, int, float]:
    free = ops.free_dofs
    if x0 is None:
        x0_free = np.zeros(len(free), dtype=float)
    else:
        x0_free = np.asarray(x0, dtype=float)[free]
    rhs_free = rhs[free]

    diagonal = np.maximum(ops.diagonal[free], 1.0e-30)

    def matvec_free(x_free: np.ndarray) -> np.ndarray:
        full = np.zeros_like(rhs)
        full[free] = x_free
        return operator_matvec(ops.diagonal, ops.i_idx, ops.j_idx, ops.values, full)[free]

    x_free, iters, rel_res = _pcg_with_matvec(matvec_free, diagonal, rhs_free, x0_free, tol, max_iter)
    full = np.zeros_like(rhs)
    full[free] = x_free
    full[ops.constrained_dofs] = 0.0
    return full, iters, rel_res


def recover_von_mises(
    mesh: AnnulusMesh,
    ops: ElasticityOperators,
    temperature_c: np.ndarray,
    displacement: np.ndarray,
    reference_temperature_c: float,
    config: dict[str, Any],
) -> np.ndarray:
    alpha = _thermal_expansion(config)
    poisson = float(config.get("thermoelastic", {}).get("material", {}).get("poisson_ratio", 0.22))
    out = np.zeros(mesh.element_count, dtype=float)
    thermal_shape = np.array([1.0, 1.0, 0.0], dtype=float)

    for elem_id, tri in enumerate(mesh.elements):
        dofs = _element_dofs(tri)
        strain = ops.element_b[elem_id] @ displacement[dofs]
        d_temp = float(np.mean(temperature_c[tri]) - reference_temperature_c)
        stress = ops.elastic_matrix @ (strain - alpha * d_temp * thermal_shape)
        sx, sy, txy = (float(stress[0]), float(stress[1]), float(stress[2]))
        if ops.model == "plane_strain":
            sz = poisson * (sx + sy) - float(config["thermoelastic"]["material"].get("young_modulus_pa", 30.0e9)) * alpha * d_temp
            vm = np.sqrt(0.5 * ((sx - sy) ** 2 + (sy - sz) ** 2 + (sz - sx) ** 2) + 3.0 * txy**2)
        else:
            vm = np.sqrt(max(0.0, sx**2 - sx * sy + sy**2 + 3.0 * txy**2))
        out[elem_id] = vm
    return out


def element_to_node_values(mesh: AnnulusMesh, ops: ElasticityOperators, element_values: np.ndarray) -> np.ndarray:
    nodal = np.zeros(mesh.node_count, dtype=float)
    weights = np.zeros(mesh.node_count, dtype=float)
    for tri, area, value in zip(mesh.elements, ops.element_area, element_values):
        np.add.at(nodal, tri, area * value / 3.0)
        np.add.at(weights, tri, area / 3.0)
    return nodal / np.maximum(weights, 1.0e-30)


def triangle_elastic_b_matrix(coords: np.ndarray) -> tuple[float, np.ndarray]:
    x0, y0 = coords[0]
    x1, y1 = coords[1]
    x2, y2 = coords[2]
    area2 = (x1 - x0) * (y2 - y0) - (y1 - y0) * (x2 - x0)
    area = 0.5 * area2
    if area <= 0.0:
        return area, np.zeros((3, 6), dtype=float)
    beta = np.array([y1 - y2, y2 - y0, y0 - y1], dtype=float) / (2.0 * area)
    gamma = np.array([x2 - x1, x0 - x2, x1 - x0], dtype=float) / (2.0 * area)
    b_mat = np.zeros((3, 6), dtype=float)
    for local in range(3):
        b_mat[0, 2 * local] = beta[local]
        b_mat[1, 2 * local + 1] = gamma[local]
        b_mat[2, 2 * local] = gamma[local]
        b_mat[2, 2 * local + 1] = beta[local]
    return area, b_mat


def rigid_body_constraints(mesh: AnnulusMesh) -> np.ndarray:
    outer = mesh.n_radial * mesh.n_theta
    node_0 = outer + _closest_angle_index(mesh, 0.0)
    node_pi = outer + _closest_angle_index(mesh, np.pi)
    node_half_pi = outer + _closest_angle_index(mesh, 0.5 * np.pi)
    dofs = [2 * node_0 + 1, 2 * node_pi + 1, 2 * node_half_pi]
    return np.asarray(sorted(set(dofs)), dtype=np.int32)


def _closest_angle_index(mesh: AnnulusMesh, angle: float) -> int:
    dist = np.abs(np.angle(np.exp(1j * (mesh.angles - angle))))
    return int(np.argmin(dist))


def _element_dofs(tri: np.ndarray) -> np.ndarray:
    dofs = np.empty(6, dtype=np.int32)
    dofs[0::2] = 2 * tri
    dofs[1::2] = 2 * tri + 1
    return dofs


def _reference_temperature(result: SimulationResult, cfg: dict[str, Any]) -> float:
    ref = cfg.get("reference_temperature", "initial_mean")
    if isinstance(ref, str):
        if ref == "initial_mean":
            return float(np.mean(result.snapshots[0]))
        if ref == "ambient":
            return float(result.config.get("cooling", {}).get("ambient_c", 25.0))
        if ref == "zero":
            return 0.0
        raise ValueError(f"Unknown thermoelastic reference_temperature: {ref}")
    return float(ref)


def _thermal_expansion(config: dict[str, Any]) -> float:
    material = config.get("thermoelastic", {}).get("material", {})
    return float(material.get("thermal_expansion", 2.0e-6))


def _is_spatially_uniform(temperature_c: np.ndarray, cfg: dict[str, Any]) -> bool:
    tolerance = float(cfg.get("uniform_temperature_tol_c", 1.0e-7))
    return float(np.max(temperature_c) - np.min(temperature_c)) <= tolerance


def _pcg_with_matvec(
    matvec,
    diagonal: np.ndarray,
    b: np.ndarray,
    x0: np.ndarray,
    tol: float,
    max_iter: int,
) -> tuple[np.ndarray, int, float]:
    # Reuse the heat solver when the operator is diagonal-only; otherwise use the
    # same preconditioned CG steps with a callback matvec for constrained DOFs.
    if len(b) == 0:
        return x0.copy(), 0, 0.0
    if matvec is None:
        return pcg_solve(diagonal, np.asarray([], dtype=np.int32), np.asarray([], dtype=np.int32), np.asarray([], dtype=float), b, x0, tol, max_iter)

    x = x0.copy()
    r = b - matvec(x)
    norm_b = float(np.linalg.norm(b))
    if norm_b == 0.0:
        return np.zeros_like(b), 0, 0.0
    rel = float(np.linalg.norm(r) / norm_b)
    if rel <= tol:
        return x, 0, rel

    inv_diag = 1.0 / np.maximum(diagonal, 1.0e-30)
    z = inv_diag * r
    p = z.copy()
    rz_old = float(np.dot(r, z))
    if rz_old == 0.0:
        return x, 0, rel

    for it in range(1, max_iter + 1):
        ap = matvec(p)
        denom = float(np.dot(p, ap))
        if abs(denom) < 1.0e-30:
            break
        alpha = rz_old / denom
        x += alpha * p
        r -= alpha * ap
        rel = float(np.linalg.norm(r) / norm_b)
        if rel <= tol:
            return x, it, rel
        z = inv_diag * r
        rz_new = float(np.dot(r, z))
        if abs(rz_old) < 1.0e-30:
            break
        beta = rz_new / rz_old
        p = z + beta * p
        rz_old = rz_new
    return x, max_iter, rel
