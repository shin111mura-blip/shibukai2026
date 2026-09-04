from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, List, Mapping, Optional, Tuple

from .schema import Edge, Node


@dataclass(frozen=True)
class SpatialThresholds:
    image_width: int = 256
    image_height: int = 256
    delta_x_fraction: float = 0.03
    delta_y_fraction: float = 0.03
    horizontal_y_alignment_fraction: float = 0.12
    vertical_x_alignment_fraction: float = 0.12
    frontback_x_alignment_fraction: float = 0.20
    frontback_depth_delta_m: float = 0.02

    @property
    def delta_x(self) -> float:
        return self.image_width * self.delta_x_fraction

    @property
    def delta_y(self) -> float:
        return self.image_height * self.delta_y_fraction

    @property
    def horizontal_y_alignment(self) -> float:
        return self.image_height * self.horizontal_y_alignment_fraction

    @property
    def vertical_x_alignment(self) -> float:
        return self.image_width * self.vertical_x_alignment_fraction

    @property
    def frontback_x_alignment(self) -> float:
        return self.image_width * self.frontback_x_alignment_fraction


def horizontal_relation(a: Node, b: Node, thresholds: SpatialThresholds) -> Optional[Edge]:
    if a.centroid_xy is None or b.centroid_xy is None:
        return None
    ax, ay = a.centroid_xy
    bx, by = b.centroid_xy
    if abs(ay - by) > thresholds.horizontal_y_alignment:
        return None
    if ax < bx - thresholds.delta_x:
        return Edge(a.id, "left_of", b.id)
    if ax > bx + thresholds.delta_x:
        return Edge(a.id, "right_of", b.id)
    return None


def vertical_relation(a: Node, b: Node, thresholds: SpatialThresholds) -> Optional[Edge]:
    if a.centroid_xy is None or b.centroid_xy is None:
        return None
    ax, ay = a.centroid_xy
    bx, by = b.centroid_xy
    if abs(ax - bx) > thresholds.vertical_x_alignment:
        return None
    if ay < by - thresholds.delta_y:
        return Edge(a.id, "above", b.id)
    if ay > by + thresholds.delta_y:
        return Edge(a.id, "below", b.id)
    return None


def spatial_edges(nodes: Iterable[Node], thresholds: SpatialThresholds, visible_only: bool) -> List[Edge]:
    selected = [node for node in nodes if node.entity_type != "gripper" and (node.visible or not visible_only)]
    edges: List[Edge] = []
    for src in selected:
        for dst in selected:
            if src.id == dst.id:
                continue
            for rule in (horizontal_relation, vertical_relation):
                edge = rule(src, dst, thresholds)
                if edge is not None:
                    edges.append(edge)
    return edges


Positions2D = Mapping[str, Tuple[float, float]]
DepthPositions = Mapping[str, float]
WorldPositions = Mapping[str, Tuple[float, float, float]]


def node_position_xy(node: Node, positions: Positions2D) -> Optional[Tuple[float, float]]:
    point = positions.get(node.id)
    if point is None:
        return None
    return float(point[0]), float(point[1])


def node_world_xy(node: Node, world_positions: WorldPositions) -> Optional[Tuple[float, float]]:
    point = world_positions.get(node.id)
    if point is None:
        return None
    return float(point[0]), float(point[1])


def relative_neighborhood_pairs(nodes: List[Node], positions: Positions2D) -> List[tuple[Node, Node]]:
    points = {node.id: node_position_xy(node, positions) for node in nodes}
    pairs: List[tuple[Node, Node]] = []
    for i, src in enumerate(nodes):
        src_point = points[src.id]
        if src_point is None:
            continue
        for dst in nodes[i + 1 :]:
            dst_point = points[dst.id]
            if dst_point is None:
                continue
            distance = math.dist(src_point, dst_point)
            blocked = False
            for other in nodes:
                if other.id in {src.id, dst.id}:
                    continue
                other_point = points[other.id]
                if other_point is None:
                    continue
                if math.dist(src_point, other_point) < distance and math.dist(dst_point, other_point) < distance:
                    blocked = True
                    break
            if not blocked:
                pairs.append((src, dst))
    return pairs


def directed_axis_edges(src: Node, dst: Node, positions: Positions2D) -> List[Edge]:
    src_point = node_position_xy(src, positions)
    dst_point = node_position_xy(dst, positions)
    if src_point is None or dst_point is None:
        return []
    sx, sy = src_point
    dx, dy = dst_point
    delta_x = dx - sx
    delta_y = dy - sy
    if abs(delta_x) >= abs(delta_y):
        if sx < dx:
            return [Edge(src.id, "left_of", dst.id), Edge(dst.id, "right_of", src.id)]
        return [Edge(src.id, "right_of", dst.id), Edge(dst.id, "left_of", src.id)]
    if sy < dy:
        return [Edge(src.id, "above", dst.id), Edge(dst.id, "below", src.id)]
    return [Edge(src.id, "below", dst.id), Edge(dst.id, "above", src.id)]


def nearest_axis_edges(nodes: List[Node], positions: Positions2D, thresholds: SpatialThresholds) -> List[Edge]:
    edges: set[tuple[str, str, str]] = set()
    points = {node.id: node_position_xy(node, positions) for node in nodes}
    for src in nodes:
        src_point = points[src.id]
        if src_point is None:
            continue
        sx, sy = src_point
        left_candidates: list[tuple[float, Node]] = []
        right_candidates: list[tuple[float, Node]] = []
        up_candidates: list[tuple[float, Node]] = []
        down_candidates: list[tuple[float, Node]] = []
        for other in nodes:
            if other.id == src.id:
                continue
            other_point = points[other.id]
            if other_point is None:
                continue
            ox, oy = other_point
            dx = ox - sx
            dy = oy - sy
            if abs(dx) >= abs(dy) and abs(dy) <= thresholds.horizontal_y_alignment:
                if ox < sx - thresholds.delta_x:
                    left_candidates.append((abs(dx), other))
                elif ox > sx + thresholds.delta_x:
                    right_candidates.append((abs(dx), other))
            if abs(dy) > abs(dx) and abs(dx) <= thresholds.vertical_x_alignment:
                if oy < sy - thresholds.delta_y:
                    up_candidates.append((abs(dy), other))
                elif oy > sy + thresholds.delta_y:
                    down_candidates.append((abs(dy), other))

        if left_candidates:
            other = min(left_candidates, key=lambda item: item[0])[1]
            edges.add((other.id, "left_of", src.id))
            edges.add((src.id, "right_of", other.id))
        if right_candidates:
            other = min(right_candidates, key=lambda item: item[0])[1]
            edges.add((src.id, "left_of", other.id))
            edges.add((other.id, "right_of", src.id))
        if up_candidates:
            other = min(up_candidates, key=lambda item: item[0])[1]
            edges.add((other.id, "above", src.id))
            edges.add((src.id, "below", other.id))
        if down_candidates:
            other = min(down_candidates, key=lambda item: item[0])[1]
            edges.add((src.id, "above", other.id))
            edges.add((other.id, "below", src.id))
    return [Edge(subject, predicate, obj) for subject, predicate, obj in sorted(edges)]


def nearest_frontback_edges(
    nodes: List[Node],
    positions: Positions2D,
    depth_positions: DepthPositions,
    thresholds: SpatialThresholds,
) -> List[Edge]:
    edges: set[tuple[str, str, str]] = set()
    points = {node.id: node_position_xy(node, positions) for node in nodes}
    for src in nodes:
        src_point = points[src.id]
        src_depth = depth_positions.get(src.id)
        if src_point is None or src_depth is None:
            continue
        sx, _sy = src_point
        front_candidates: list[tuple[float, Node]] = []
        back_candidates: list[tuple[float, Node]] = []
        for other in nodes:
            if other.id == src.id:
                continue
            other_point = points[other.id]
            other_depth = depth_positions.get(other.id)
            if other_point is None or other_depth is None:
                continue
            ox, _oy = other_point
            if abs(ox - sx) > thresholds.frontback_x_alignment:
                continue
            delta_depth = float(other_depth) - float(src_depth)
            if delta_depth < -thresholds.frontback_depth_delta_m:
                front_candidates.append((abs(delta_depth), other))
            elif delta_depth > thresholds.frontback_depth_delta_m:
                back_candidates.append((abs(delta_depth), other))

        if front_candidates:
            other = min(front_candidates, key=lambda item: item[0])[1]
            edges.add((other.id, "front_of", src.id))
            edges.add((src.id, "behind", other.id))
        if back_candidates:
            other = min(back_candidates, key=lambda item: item[0])[1]
            edges.add((src.id, "front_of", other.id))
            edges.add((other.id, "behind", src.id))
    return [Edge(subject, predicate, obj) for subject, predicate, obj in sorted(edges)]


def structural_spatial_edges(
    nodes: Iterable[Node],
    positions: Positions2D,
    depth_positions: DepthPositions | None = None,
    visible_only: bool = False,
    thresholds: SpatialThresholds | None = None,
) -> List[Edge]:
    thresholds = thresholds or SpatialThresholds()
    selected = [
        node
        for node in nodes
        if node.entity_type != "gripper" and node.present and (node.visible or not visible_only) and node_position_xy(node, positions) is not None
    ]
    edges = nearest_axis_edges(selected, positions, thresholds)
    if depth_positions:
        edges.extend(nearest_frontback_edges(selected, positions, depth_positions, thresholds))
    return edges


def observable_subset(nodes: Iterable[Node], edges: Iterable[Edge]) -> tuple[List[Node], List[Edge]]:
    visible_nodes = [node for node in nodes if node.visible]
    visible_ids = {node.id for node in visible_nodes}
    visible_edges = [edge for edge in edges if edge.subject in visible_ids and edge.object in visible_ids]
    return visible_nodes, visible_edges
