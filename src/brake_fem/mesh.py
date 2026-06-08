from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AnnulusMesh:
    nodes: np.ndarray
    elements: np.ndarray
    radii: np.ndarray
    angles: np.ndarray
    node_r: np.ndarray
    node_theta: np.ndarray
    inner_boundary_edges: np.ndarray
    outer_boundary_edges: np.ndarray
    n_radial: int
    n_theta: int

    @property
    def node_count(self) -> int:
        return int(self.nodes.shape[0])

    @property
    def element_count(self) -> int:
        return int(self.elements.shape[0])


def generate_annulus_mesh(
    r_inner: float,
    r_outer: float,
    n_radial: int,
    n_theta: int,
) -> AnnulusMesh:
    if r_inner <= 0.0 or r_outer <= r_inner:
        raise ValueError("Annulus radii must satisfy 0 < r_inner < r_outer.")
    if n_radial < 1 or n_theta < 8:
        raise ValueError("Use at least one radial cell and eight angular cells.")

    radii = np.linspace(r_inner, r_outer, n_radial + 1)
    angles = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False)

    nodes = []
    node_r = []
    node_theta = []
    for i, radius in enumerate(radii):
        for j, theta in enumerate(angles):
            nodes.append([radius * np.cos(theta), radius * np.sin(theta)])
            node_r.append(radius)
            node_theta.append(theta)
    nodes_arr = np.asarray(nodes, dtype=float)
    node_r_arr = np.asarray(node_r, dtype=float)
    node_theta_arr = np.asarray(node_theta, dtype=float)

    elements = []
    for i in range(n_radial):
        for j in range(n_theta):
            j2 = (j + 1) % n_theta
            n00 = _node_id(i, j, n_theta)
            n01 = _node_id(i, j2, n_theta)
            n10 = _node_id(i + 1, j, n_theta)
            n11 = _node_id(i + 1, j2, n_theta)
            tri1 = [n00, n10, n11]
            tri2 = [n00, n11, n01]
            elements.append(_positive_orientation(tri1, nodes_arr))
            elements.append(_positive_orientation(tri2, nodes_arr))

    inner_edges = []
    outer_edges = []
    for j in range(n_theta):
        j2 = (j + 1) % n_theta
        inner_edges.append([_node_id(0, j, n_theta), _node_id(0, j2, n_theta)])
        outer_edges.append([_node_id(n_radial, j, n_theta), _node_id(n_radial, j2, n_theta)])

    return AnnulusMesh(
        nodes=nodes_arr,
        elements=np.asarray(elements, dtype=np.int32),
        radii=radii,
        angles=angles,
        node_r=node_r_arr,
        node_theta=node_theta_arr,
        inner_boundary_edges=np.asarray(inner_edges, dtype=np.int32),
        outer_boundary_edges=np.asarray(outer_edges, dtype=np.int32),
        n_radial=int(n_radial),
        n_theta=int(n_theta),
    )


def triangle_areas(mesh: AnnulusMesh) -> np.ndarray:
    pts = mesh.nodes[mesh.elements]
    cross = (
        (pts[:, 1, 0] - pts[:, 0, 0]) * (pts[:, 2, 1] - pts[:, 0, 1])
        - (pts[:, 1, 1] - pts[:, 0, 1]) * (pts[:, 2, 0] - pts[:, 0, 0])
    )
    return 0.5 * cross


def mesh_area(mesh: AnnulusMesh) -> float:
    return float(np.sum(triangle_areas(mesh)))


def _node_id(i: int, j: int, n_theta: int) -> int:
    return i * n_theta + j


def _positive_orientation(nodes: list[int], coords: np.ndarray) -> list[int]:
    p = coords[nodes]
    area2 = (p[1, 0] - p[0, 0]) * (p[2, 1] - p[0, 1]) - (
        p[1, 1] - p[0, 1]
    ) * (p[2, 0] - p[0, 0])
    if area2 < 0.0:
        return [nodes[0], nodes[2], nodes[1]]
    return nodes
