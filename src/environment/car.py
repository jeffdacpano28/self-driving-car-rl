"""
Car class with realistic physics for the self-driving car simulation.
"""

import math
import numpy as np
from typing import List, Tuple, TYPE_CHECKING, Optional

try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False
    if TYPE_CHECKING:
        import pygame

if TYPE_CHECKING:
    from .sensor import SensorArray
    from .track import Track


class Car:
    """
    Represents a car with realistic physics including:
    - Position and orientation
    - Velocity and acceleration
    - Steering mechanics
    - Collision detection via corner points
    """

    # Action constants
    TURN_LEFT = 0
    STRAIGHT = 1
    TURN_RIGHT = 2

    def __init__(
        self,
        x: float,
        y: float,
        angle: float = 0.0,
        width: float = 30.0,
        height: float = 50.0,
        max_velocity: float = 10.0,
        min_velocity: float = -3.0,
        acceleration: float = 0.5,
        friction: float = 0.98,
        turn_rate: float = 0.1,
    ):
        """
        Initialize a car.

        Args:
            x: Initial x position
            y: Initial y position
            angle: Initial heading angle in radians (0 = right, π/2 = down)
            width: Car width in pixels
            height: Car length in pixels
            max_velocity: Maximum forward speed
            min_velocity: Maximum reverse speed (negative value)
            acceleration: Acceleration rate per frame
            friction: Velocity decay factor (< 1.0)
            turn_rate: Steering angle change per frame in radians
        """
        # Position and orientation
        self.x = x
        self.y = y
        self.angle = angle  # Radians

        # Dimensions
        self.width = width
        self.height = height

        # Movement state
        self.velocity = 0.0
        self.acceleration_rate = acceleration
        self.friction = friction

        # Physics limits
        self.max_velocity = max_velocity
        self.min_velocity = min_velocity
        self.turn_rate = turn_rate

        # Track if car is alive
        self.alive = True

        # Sensors (optional, attach later)
        self.sensors: Optional["SensorArray"] = None

        # Store initial state for reset
        self.initial_x = x
        self.initial_y = y
        self.initial_angle = angle

    def update(self, action: int) -> None:
        """
        Update car state based on action.

        Args:
            action: 0=TURN_LEFT, 1=STRAIGHT, 2=TURN_RIGHT
        """
        if not self.alive:
            return

        # Always accelerate forward
        self.velocity += self.acceleration_rate

        # Apply steering
        if action == self.TURN_LEFT:
            self.angle -= self.turn_rate
        elif action == self.TURN_RIGHT:
            self.angle += self.turn_rate
        # STRAIGHT: no angle change

        # Clamp velocity
        self.velocity = max(self.min_velocity, min(self.max_velocity, self.velocity))

        # Apply friction
        self.velocity *= self.friction

        # Update position based on velocity and angle
        self.x += math.cos(self.angle) * self.velocity
        self.y += math.sin(self.angle) * self.velocity

    def get_corners(self) -> List[Tuple[float, float]]:
        """
        Get the 4 corner positions of the car for collision detection.

        Returns:
            List of (x, y) tuples representing the 4 corners
        """
        # Half dimensions
        hw = self.width / 2
        hh = self.height / 2

        # Local corners (relative to car center, before rotation)
        local_corners = [
            (-hw, -hh),  # Top-left
            (hw, -hh),  # Top-right
            (hw, hh),  # Bottom-right
            (-hw, hh),  # Bottom-left
        ]

        # Rotate and translate to world coordinates
        corners = []
        cos_a = math.cos(self.angle)
        sin_a = math.sin(self.angle)

        for lx, ly in local_corners:
            # Rotate
            rx = lx * cos_a - ly * sin_a
            ry = lx * sin_a + ly * cos_a
            # Translate
            wx = self.x + rx
            wy = self.y + ry
            corners.append((wx, wy))

        return corners

    def get_front_center(self) -> Tuple[float, float]:
        """
        Get the position of the front center of the car.
        Useful for sensor placement.

        Returns:
            (x, y) tuple of front center position
        """
        front_offset = self.height / 2
        fx = self.x + math.cos(self.angle) * front_offset
        fy = self.y + math.sin(self.angle) * front_offset
        return (fx, fy)

    def attach_sensors(self, sensors: "SensorArray") -> None:
        """
        Attach a sensor array to the car.

        Args:
            sensors: SensorArray instance
        """
        self.sensors = sensors

    def read_sensors(self, track: "Track") -> np.ndarray:
        """
        Read all sensors and return normalized distances.

        Args:
            track: Track to check collision with

        Returns:
            NumPy array of normalized sensor readings [0.0-1.0]
            Returns empty array if no sensors attached
        """
        if self.sensors is None:
            return np.array([], dtype=np.float32)

        return self.sensors.read_all(self.x, self.y, self.angle, track)

    def get_observation(self, track: "Track") -> np.ndarray:
        """
        Get full observation vector for RL agent.
        Combines sensor readings with velocity.

        Args:
            track: Track to read sensors against

        Returns:
            NumPy array: [sensor1, sensor2, ..., sensorN, velocity_normalized]
        """
        # Read sensors
        sensor_readings = self.read_sensors(track)

        # Normalize velocity to [0, 1]
        velocity_norm = (self.velocity - self.min_velocity) / (
            self.max_velocity - self.min_velocity
        )
        velocity_norm = np.clip(velocity_norm, 0.0, 1.0)

        # Combine sensor readings with velocity
        observation = np.append(sensor_readings, velocity_norm)

        return observation.astype(np.float32)

    def reset(self) -> None:
        """Reset car to initial state."""
        self.x = self.initial_x
        self.y = self.initial_y
        self.angle = self.initial_angle
        self.velocity = 0.0
        self.alive = True

    def kill(self) -> None:
        """Mark car as dead (crashed)."""
        self.alive = False
        self.velocity = 0.0

    def get_state_vector(self) -> np.ndarray:
        """
        Get car state as a vector (useful for debugging/logging).

        Returns:
            np.array: [x, y, angle, velocity]
        """
        return np.array([self.x, self.y, self.angle, self.velocity])

    def render(
        self,
        screen: Optional["pygame.Surface"] = None,
        color: Tuple[int, int, int] = (66, 135, 245),
        show_sensors: bool = True,
    ) -> None:
        """
        Render the car on a pygame surface.

        Args:
            screen: Pygame surface to draw on (optional)
            color: RGB color tuple
            show_sensors: Whether to render sensor rays
        """
        if not PYGAME_AVAILABLE or screen is None:
            return

        # Render sensors first (so they appear behind car)
        if show_sensors and self.sensors is not None:
            self.sensors.render(screen, self.x, self.y)

        if not self.alive:
            color = (150, 150, 150)  # Gray if dead

        corners = self.get_corners()
        pygame.draw.polygon(screen, color, corners)

        # Draw a line indicating direction (front of car)
        front_center = self.get_front_center()
        pygame.draw.line(screen, (255, 255, 0), (self.x, self.y), front_center, 3)

    def __repr__(self) -> str:
        return (
            f"Car(x={self.x:.1f}, y={self.y:.1f}, "
            f"angle={math.degrees(self.angle):.1f}°, "
            f"velocity={self.velocity:.2f}, alive={self.alive})"
        )
