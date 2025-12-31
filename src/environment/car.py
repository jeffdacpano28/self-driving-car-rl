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
    - Smooth steering mechanics (angular velocity)
    - Collision detection via corner points
    - Pure self-learning mode (NO intelligent assists)
    """

    # Action constants (9 discrete actions = 3 steering × 3 speed)
    LEFT_SLOW = 0
    LEFT_NORMAL = 1
    LEFT_FAST = 2
    STRAIGHT_SLOW = 3
    STRAIGHT_NORMAL = 4
    STRAIGHT_FAST = 5
    RIGHT_SLOW = 6
    RIGHT_NORMAL = 7
    RIGHT_FAST = 8

    # Speed multipliers for SLOW/NORMAL/FAST
    SPEED_SLOW = 0.3      # 30% of max velocity
    SPEED_NORMAL = 0.6    # 60% of max velocity
    SPEED_FAST = 1.0      # 100% of max velocity

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
        angular_damping: float = 0.85,
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
            turn_rate: Steering angular velocity per frame in radians
            angular_damping: Damping factor for smooth steering (0.0=instant, 1.0=no damping)
        """
        # Position and orientation
        self.x = x
        self.y = y
        self.angle = angle  # Radians
        self.angular_velocity = 0.0  # Current turning speed

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
        self.angular_damping = angular_damping

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
        Update car state based on 9 discrete actions (3 steering × 3 speed).
        Pure self-learning mode - NO intelligent assists!
        Features smooth steering using angular velocity.

        Args:
            action: 0-8 representing [LEFT/STRAIGHT/RIGHT] × [SLOW/NORMAL/FAST]
                0: LEFT_SLOW, 1: LEFT_NORMAL, 2: LEFT_FAST
                3: STRAIGHT_SLOW, 4: STRAIGHT_NORMAL, 5: STRAIGHT_FAST
                6: RIGHT_SLOW, 7: RIGHT_NORMAL, 8: RIGHT_FAST
        """
        if not self.alive:
            return

        # Decode action into steering and speed
        # action // 3 gives steering: 0=LEFT, 1=STRAIGHT, 2=RIGHT
        # action % 3 gives speed: 0=SLOW, 1=NORMAL, 2=FAST
        steering = action // 3
        speed_level = action % 3

        # Map speed level to target speed multiplier
        if speed_level == 0:  # SLOW
            speed_multiplier = self.SPEED_SLOW
        elif speed_level == 1:  # NORMAL
            speed_multiplier = self.SPEED_NORMAL
        else:  # FAST
            speed_multiplier = self.SPEED_FAST

        # Calculate target speed
        target_speed = self.max_velocity * speed_multiplier

        # Smooth acceleration/deceleration toward target speed
        speed_diff = target_speed - self.velocity
        if abs(speed_diff) > 0.1:
            # Accelerate or brake
            self.velocity += speed_diff * 0.1  # Smooth transition
        else:
            self.velocity = target_speed

        # Smooth steering using angular velocity
        # Map steering to target angular velocity
        if steering == 0:  # LEFT
            target_angular_velocity = -self.turn_rate
        elif steering == 2:  # RIGHT
            target_angular_velocity = self.turn_rate
        else:  # STRAIGHT
            target_angular_velocity = 0.0

        # Apply damping to angular velocity (smooth steering)
        self.angular_velocity += (target_angular_velocity - self.angular_velocity) * (1.0 - self.angular_damping)

        # Update angle with angular velocity
        self.angle += self.angular_velocity

        # Clamp velocity to limits
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
