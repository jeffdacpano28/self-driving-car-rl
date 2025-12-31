"""
Track class for defining racing circuits with collision detection.
"""

import json
import math
from typing import List, Tuple, Dict, Optional, TYPE_CHECKING
import numpy as np

try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False
    if TYPE_CHECKING:
        import pygame


class Line:
    """Represents a line segment (used for checkpoints)."""

    def __init__(self, start: Tuple[float, float], end: Tuple[float, float]):
        """
        Initialize a line segment.

        Args:
            start: (x, y) start point
            end: (x, y) end point
        """
        self.start = start
        self.end = end

    def intersects(self, p1: Tuple[float, float], p2: Tuple[float, float]) -> bool:
        """
        Check if line segment (p1, p2) intersects with this line.
        Uses cross product method.

        Args:
            p1: First point of segment to check
            p2: Second point of segment to check

        Returns:
            True if segments intersect
        """
        def ccw(A, B, C):
            """Check if three points are in counter-clockwise order."""
            return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

        A, B = self.start, self.end
        C, D = p1, p2

        return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)

    def __repr__(self) -> str:
        return f"Line({self.start} -> {self.end})"


class Track:
    """
    Represents a racing track with inner and outer boundaries.
    Uses polygon-based collision detection.
    """

    def __init__(
        self,
        name: str,
        outer_boundary: List[Tuple[float, float]],
        inner_boundary: List[Tuple[float, float]],
        checkpoints: List[Dict[str, Tuple[float, float]]],
        start_pos: Tuple[float, float],
        start_angle: float = 0.0,
    ):
        """
        Initialize a track.

        Args:
            name: Track name
            outer_boundary: List of (x, y) points defining outer wall
            inner_boundary: List of (x, y) points defining inner wall
            checkpoints: List of dicts with 'start' and 'end' keys
            start_pos: (x, y) starting position for car
            start_angle: Starting angle in radians
        """
        self.name = name
        self.outer_boundary = outer_boundary
        self.inner_boundary = inner_boundary
        self.start_pos = start_pos
        self.start_angle = start_angle

        # Convert checkpoint dicts to Line objects
        self.checkpoints = [
            Line(cp["start"], cp["end"]) for cp in checkpoints
        ]

        # Track dimensions for rendering
        self._calculate_bounds()

    def _calculate_bounds(self) -> None:
        """Calculate bounding box of the track."""
        all_points = self.outer_boundary + self.inner_boundary

        xs = [p[0] for p in all_points]
        ys = [p[1] for p in all_points]

        self.min_x = min(xs)
        self.max_x = max(xs)
        self.min_y = min(ys)
        self.max_y = max(ys)
        self.width = self.max_x - self.min_x
        self.height = self.max_y - self.min_y

    def is_point_on_track(self, x: float, y: float) -> bool:
        """
        Check if a point is on the track (between inner and outer boundaries).
        Uses ray-casting algorithm for point-in-polygon test.

        Args:
            x: X coordinate
            y: Y coordinate

        Returns:
            True if point is on track (inside outer AND outside inner)
        """
        # Must be inside outer boundary
        inside_outer = self._point_in_polygon(x, y, self.outer_boundary)

        # If no inner boundary, just check outer
        if not self.inner_boundary:
            return inside_outer

        # Must also be outside inner boundary
        inside_inner = self._point_in_polygon(x, y, self.inner_boundary)
        return inside_outer and not inside_inner

    def _point_in_polygon(self, x: float, y: float, polygon: List[Tuple[float, float]]) -> bool:
        """
        Ray-casting algorithm to check if point is inside polygon.

        Args:
            x: X coordinate
            y: Y coordinate
            polygon: List of (x, y) points defining polygon

        Returns:
            True if point is inside polygon
        """
        n = len(polygon)
        inside = False

        p1x, p1y = polygon[0]
        for i in range(1, n + 1):
            p2x, p2y = polygon[i % n]

            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside

            p1x, p1y = p2x, p2y

        return inside

    def check_checkpoint(
        self,
        prev_pos: Tuple[float, float],
        curr_pos: Tuple[float, float],
        checkpoint_idx: int,
    ) -> bool:
        """
        Check if car crossed a checkpoint.

        Args:
            prev_pos: Previous car position
            curr_pos: Current car position
            checkpoint_idx: Index of checkpoint to check

        Returns:
            True if car crossed the checkpoint
        """
        if checkpoint_idx >= len(self.checkpoints):
            return False

        checkpoint = self.checkpoints[checkpoint_idx]
        return checkpoint.intersects(prev_pos, curr_pos)

    def get_next_checkpoint_idx(self, current_idx: int) -> int:
        """
        Get index of next checkpoint.

        Args:
            current_idx: Current checkpoint index

        Returns:
            Next checkpoint index (wraps around)
        """
        return (current_idx + 1) % len(self.checkpoints)

    def render(
        self,
        screen: Optional["pygame.Surface"] = None,
        show_checkpoints: bool = True,
    ) -> None:
        """
        Render the track on a pygame surface.

        Args:
            screen: Pygame surface to draw on
            show_checkpoints: Whether to draw checkpoint lines
        """
        if not PYGAME_AVAILABLE or screen is None:
            return

        # Draw outer boundary (dark gray)
        pygame.draw.polygon(screen, (50, 50, 50), self.outer_boundary)

        # Draw inner boundary (background color - creates "hole") if exists
        if self.inner_boundary and len(self.inner_boundary) > 2:
            pygame.draw.polygon(screen, (34, 34, 34), self.inner_boundary)

        # Draw boundary lines
        pygame.draw.lines(screen, (200, 200, 200), True, self.outer_boundary, 3)
        if self.inner_boundary and len(self.inner_boundary) > 2:
            pygame.draw.lines(screen, (200, 200, 200), True, self.inner_boundary, 3)

        # Draw checkpoints
        if show_checkpoints:
            for i, checkpoint in enumerate(self.checkpoints):
                color = (156, 39, 176)  # Purple
                pygame.draw.line(
                    screen,
                    color,
                    checkpoint.start,
                    checkpoint.end,
                    2,
                )
                # Draw checkpoint number
                if hasattr(pygame, 'font') and pygame.font.get_init():
                    font = pygame.font.Font(None, 24)
                    text = font.render(str(i), True, color)
                    mid_x = (checkpoint.start[0] + checkpoint.end[0]) / 2
                    mid_y = (checkpoint.start[1] + checkpoint.end[1]) / 2
                    screen.blit(text, (mid_x - 10, mid_y - 10))

        # Draw start position (green circle)
        pygame.draw.circle(screen, (76, 175, 80), self.start_pos, 10)

    @classmethod
    def load(cls, filepath: str) -> "Track":
        """
        Load a track from a JSON file.

        Args:
            filepath: Path to JSON file

        Returns:
            Track instance

        Example JSON format:
        {
            "name": "Oval Easy",
            "outer_boundary": [[x1, y1], [x2, y2], ...],
            "inner_boundary": [[x1, y1], [x2, y2], ...],
            "checkpoints": [
                {"start": [x1, y1], "end": [x2, y2]},
                ...
            ],
            "start_pos": [x, y],
            "start_angle": 0.0
        }
        """
        with open(filepath, "r") as f:
            data = json.load(f)

        return cls(
            name=data["name"],
            outer_boundary=[tuple(p) for p in data["outer_boundary"]],
            inner_boundary=[tuple(p) for p in data["inner_boundary"]],
            checkpoints=data["checkpoints"],
            start_pos=tuple(data["start_pos"]),
            start_angle=data.get("start_angle", 0.0),
        )

    def save(self, filepath: str) -> None:
        """
        Save track to a JSON file.

        Args:
            filepath: Path to save JSON file
        """
        data = {
            "name": self.name,
            "outer_boundary": self.outer_boundary,
            "inner_boundary": self.inner_boundary,
            "checkpoints": [
                {"start": list(cp.start), "end": list(cp.end)}
                for cp in self.checkpoints
            ],
            "start_pos": list(self.start_pos),
            "start_angle": self.start_angle,
        }

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

    def __repr__(self) -> str:
        return (
            f"Track(name='{self.name}', "
            f"outer_points={len(self.outer_boundary)}, "
            f"inner_points={len(self.inner_boundary)}, "
            f"checkpoints={len(self.checkpoints)})"
        )
