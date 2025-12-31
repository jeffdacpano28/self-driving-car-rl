"""
Ray-casting sensor system for the self-driving car.
"""

import math
from typing import List, Tuple, Optional, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from .car import Car
    from .track import Track
    try:
        import pygame
    except ImportError:
        pass


class RaySensor:
    """
    A single ray-casting sensor that detects distance to obstacles.
    """

    def __init__(
        self,
        angle_offset: float,
        max_range: float = 200.0,
    ):
        """
        Initialize a ray sensor.

        Args:
            angle_offset: Angle offset from car heading in degrees
            max_range: Maximum detection range in pixels
        """
        self.angle_offset = math.radians(angle_offset)  # Convert to radians
        self.max_range = max_range
        self.current_distance = max_range  # Current reading
        self.hit_point: Optional[Tuple[float, float]] = None  # Where ray hit

    def cast(
        self,
        car_x: float,
        car_y: float,
        car_angle: float,
        track: "Track",
        step_size: float = 2.0,
    ) -> float:
        """
        Cast a ray from car position and find distance to track boundary.

        Args:
            car_x: Car x position
            car_y: Car y position
            car_angle: Car heading angle in radians
            track: Track to check collision with
            step_size: Step size for ray marching in pixels

        Returns:
            Normalized distance [0.0 = at wall, 1.0 = no obstacle in range]
        """
        # Calculate absolute ray angle
        ray_angle = car_angle + self.angle_offset

        # Direction vector
        dx = math.cos(ray_angle)
        dy = math.sin(ray_angle)

        # Start from car center
        x, y = car_x, car_y

        # March along ray until hitting boundary or max range
        distance = 0.0
        self.hit_point = None

        while distance < self.max_range:
            # Move along ray
            x += dx * step_size
            y += dy * step_size
            distance += step_size

            # Check if point is off track (hit boundary)
            if not track.is_point_on_track(x, y):
                self.hit_point = (x, y)
                self.current_distance = distance
                # Normalize: 0.0 = at wall, 1.0 = max range
                return 1.0 - (distance / self.max_range)

        # No hit within range
        self.current_distance = self.max_range
        self.hit_point = (x, y)
        return 0.0  # Maximum distance = 0.0 (safest)

    def get_normalized_reading(self) -> float:
        """
        Get current normalized distance reading.

        Returns:
            Normalized distance [0.0 = max range, 1.0 = at wall]
        """
        return 1.0 - (self.current_distance / self.max_range)

    def get_raw_distance(self) -> float:
        """
        Get current raw distance in pixels.

        Returns:
            Distance in pixels
        """
        return self.current_distance


class SensorArray:
    """
    Array of multiple ray sensors for the car.
    """

    def __init__(
        self,
        num_sensors: int = 5,
        angles: Optional[List[float]] = None,
        max_range: float = 200.0,
    ):
        """
        Initialize sensor array.

        Args:
            num_sensors: Number of sensors
            angles: List of angle offsets in degrees (if None, use default spread)
            max_range: Maximum detection range in pixels
        """
        self.num_sensors = num_sensors
        self.max_range = max_range

        # Default angles: evenly spread from -60 to +60 degrees
        if angles is None:
            if num_sensors == 1:
                angles = [0]
            else:
                # Spread from -60 to +60
                angles = [
                    -60 + (120 / (num_sensors - 1)) * i
                    for i in range(num_sensors)
                ]

        self.angles = angles

        # Create sensors
        self.sensors = [
            RaySensor(angle, max_range) for angle in angles
        ]

    def read_all(
        self,
        car_x: float,
        car_y: float,
        car_angle: float,
        track: "Track",
    ) -> np.ndarray:
        """
        Read all sensors and return normalized distances.

        Args:
            car_x: Car x position
            car_y: Car y position
            car_angle: Car heading angle in radians
            track: Track to check collision with

        Returns:
            NumPy array of normalized distances [0.0-1.0]
        """
        readings = []
        for sensor in self.sensors:
            reading = sensor.cast(car_x, car_y, car_angle, track)
            readings.append(reading)

        return np.array(readings, dtype=np.float32)

    def render(
        self,
        screen: "pygame.Surface",
        car_x: float,
        car_y: float,
    ) -> None:
        """
        Render sensor rays on screen with color coding.

        Args:
            screen: Pygame surface to draw on
            car_x: Car x position
            car_y: Car y position
        """
        try:
            import pygame
        except ImportError:
            return

        for sensor in self.sensors:
            if sensor.hit_point is None:
                continue

            # Color based on distance (green=safe, yellow=warn, red=danger)
            normalized = sensor.get_normalized_reading()

            if normalized < 0.2:
                # Safe (far from wall)
                color = (76, 175, 80)  # Green
            elif normalized < 0.8:
                # Warning (getting close)
                color = (255, 193, 7)  # Yellow
            else:
                # Danger (very close to wall)
                color = (244, 67, 54)  # Red

            # Draw line from car to hit point
            pygame.draw.line(
                screen,
                color,
                (int(car_x), int(car_y)),
                (int(sensor.hit_point[0]), int(sensor.hit_point[1])),
                2,
            )

            # Draw small circle at hit point
            pygame.draw.circle(
                screen,
                color,
                (int(sensor.hit_point[0]), int(sensor.hit_point[1])),
                4,
            )

    def get_readings_array(self) -> np.ndarray:
        """
        Get current normalized readings from all sensors.

        Returns:
            NumPy array of normalized distances
        """
        return np.array(
            [sensor.get_normalized_reading() for sensor in self.sensors],
            dtype=np.float32,
        )

    def __repr__(self) -> str:
        readings = self.get_readings_array()
        return f"SensorArray({self.num_sensors} sensors, readings={readings})"


# Helper function for easy sensor creation
def create_default_sensors(max_range: float = 200.0) -> SensorArray:
    """
    Create default 7-sensor array with wide coverage.

    Args:
        max_range: Maximum detection range

    Returns:
        SensorArray with 7 sensors at -90, -60, -30, 0, +30, +60, +90 degrees
    """
    return SensorArray(
        num_sensors=7,
        angles=[-90, -60, -30, 0, 30, 60, 90],
        max_range=max_range,
    )
