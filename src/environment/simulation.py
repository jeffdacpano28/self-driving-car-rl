"""
Main simulation environment for the self-driving car.
Provides a Gym-like interface for reinforcement learning.
"""

from typing import Dict, Tuple, Optional, Any
import numpy as np
from pathlib import Path
import time

try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False

from .car import Car
from .track import Track
from .sensor import create_default_sensors, SensorArray
from utils.config_loader import (
    load_environment_config,
    get_car_params_from_config,
    get_sensor_params_from_config,
    get_reward_params_from_config,
    get_finish_time_params_from_config,
    get_episode_params_from_config,
)


class CarEnvironment:
    """
    Self-driving car environment with OpenAI Gym-like interface.

    Observation Space: [sensor1, sensor2, sensor3, sensor4, sensor5, velocity]
        - 5 ray-cast sensors (normalized distances [0-1])
        - 1 velocity (normalized [0-1])

    Action Space: Discrete(3)
        - 0: TURN_LEFT
        - 1: STRAIGHT
        - 2: TURN_RIGHT

    Reward:
        - +1.0 per step survived
        - +100.0 per checkpoint passed
        - -100.0 on crash
    """

    def __init__(
        self,
        track_path: str,
        render_mode: Optional[str] = None,
        config_path: Optional[str] = None,
        # Override parameters (if None, uses config)
        max_steps: Optional[int] = None,
        sensor_range: Optional[float] = None,
        num_sensors: Optional[int] = None,
        sensor_angles: Optional[list] = None,
        reward_checkpoint: Optional[float] = None,
        reward_survival: Optional[float] = None,
        reward_crash: Optional[float] = None,
        car_width: Optional[float] = None,
        car_height: Optional[float] = None,
        car_max_velocity: Optional[float] = None,
        car_min_velocity: Optional[float] = None,
        car_acceleration: Optional[float] = None,
        car_friction: Optional[float] = None,
        car_turn_rate: Optional[float] = None,
    ):
        """
        Initialize the environment.

        All parameters are optional and will be loaded from config/environment.yaml by default.
        You can override specific parameters by passing them explicitly.

        Args:
            track_path: Path to track JSON file
            render_mode: "human" for pygame visualization, None for no rendering
            config_path: Path to environment config YAML (default: config/environment.yaml)
            max_steps: Maximum steps per episode (overrides config)
            sensor_range: Maximum sensor detection range (overrides config)
            num_sensors: Number of sensors (overrides config)
            sensor_angles: Sensor angles in degrees (overrides config)
            reward_checkpoint: Reward for passing checkpoint (overrides config)
            reward_survival: Reward per step survived (overrides config)
            reward_crash: Penalty for crashing (overrides config)
            car_width: Car width in pixels (overrides config)
            car_height: Car height in pixels (overrides config)
            car_max_velocity: Maximum forward speed (overrides config)
            car_min_velocity: Maximum reverse speed (overrides config)
            car_acceleration: Acceleration rate (overrides config)
            car_friction: Velocity decay factor (overrides config)
            car_turn_rate: Steering angle change rate (overrides config)
        """
        # Load configuration
        env_config = load_environment_config(config_path)

        # Extract parameters from config
        car_params = get_car_params_from_config(env_config)
        sensor_params = get_sensor_params_from_config(env_config)
        reward_params = get_reward_params_from_config(env_config)
        finish_time_params = get_finish_time_params_from_config(env_config)
        episode_params = get_episode_params_from_config(env_config)

        # Use config values, but allow overrides
        final_car_width = car_width if car_width is not None else car_params['car_width']
        final_car_height = car_height if car_height is not None else car_params['car_height']
        final_car_max_velocity = car_max_velocity if car_max_velocity is not None else car_params['car_max_velocity']
        final_car_min_velocity = car_min_velocity if car_min_velocity is not None else car_params['car_min_velocity']
        final_car_acceleration = car_acceleration if car_acceleration is not None else car_params['car_acceleration']
        final_car_friction = car_friction if car_friction is not None else car_params['car_friction']
        final_car_turn_rate = car_turn_rate if car_turn_rate is not None else car_params['car_turn_rate']

        final_num_sensors = num_sensors if num_sensors is not None else sensor_params['num_sensors']
        final_sensor_angles = sensor_angles if sensor_angles is not None else sensor_params['angles']
        final_sensor_range = sensor_range if sensor_range is not None else sensor_params['max_range']

        final_reward_survival = reward_survival if reward_survival is not None else reward_params['reward_survival']
        final_reward_checkpoint = reward_checkpoint if reward_checkpoint is not None else reward_params['reward_checkpoint']
        final_reward_crash = reward_crash if reward_crash is not None else reward_params['reward_crash']

        final_max_steps = max_steps if max_steps is not None else episode_params['max_steps']

        # Load track
        self.track = Track.load(track_path)

        # Initialize car at track start position
        self.car = Car(
            x=self.track.start_pos[0],
            y=self.track.start_pos[1],
            angle=self.track.start_angle,
            width=final_car_width,
            height=final_car_height,
            max_velocity=final_car_max_velocity,
            min_velocity=final_car_min_velocity,
            acceleration=final_car_acceleration,
            friction=final_car_friction,
            turn_rate=final_car_turn_rate,
        )

        # Attach sensors
        self.sensors = create_default_sensors(
            num_sensors=final_num_sensors,
            angles=final_sensor_angles,
            max_range=final_sensor_range
        )
        self.car.attach_sensors(self.sensors)

        # Store sensor count for observation space
        self.num_sensors = final_num_sensors

        # Environment settings
        self.max_steps = final_max_steps
        self.render_mode = render_mode

        # Reward parameters
        self.reward_checkpoint = final_reward_checkpoint
        self.reward_survival = final_reward_survival
        self.reward_crash = final_reward_crash
        self.reward_finish = reward_params['reward_finish']

        # Finish time bonus parameters
        self.finish_time_enabled = finish_time_params['finish_time_enabled']
        self.finish_time_mode = finish_time_params['finish_time_mode']
        # Speed-based mode
        self.finish_time_max_time = finish_time_params['finish_time_max_time']
        self.finish_time_speed_multiplier = finish_time_params['finish_time_speed_multiplier']
        # Target-based mode
        self.finish_time_target = finish_time_params['finish_time_target']
        self.finish_time_bonus_multiplier = finish_time_params['finish_time_bonus_multiplier']
        self.finish_time_penalty_multiplier = finish_time_params['finish_time_penalty_multiplier']

        # Episode state
        self.current_step = 0
        self.current_checkpoint = 0
        self.total_reward = 0.0
        self.prev_pos = (self.car.x, self.car.y)

        # Position trail for visualization
        self.position_trail = []
        self.max_trail_length = 100  # Keep last 100 positions

        # Rendering (Pygame)
        self.screen = None
        self.clock = None
        self.window_width = 1000
        self.window_height = 700
        self.fps = 60

        if self.render_mode == "human" and PYGAME_AVAILABLE:
            self._init_rendering()

    def _init_rendering(self) -> None:
        """Initialize Pygame rendering."""
        if not PYGAME_AVAILABLE:
            return

        pygame.init()
        self.screen = pygame.display.set_mode(
            (self.window_width, self.window_height),
            pygame.RESIZABLE
        )
        pygame.display.set_caption("Self-Driving Car Simulation")
        self.clock = pygame.time.Clock()

        # Initialize font for info text
        pygame.font.init()
        self.font = pygame.font.Font(None, 28)

    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        """
        Reset the environment to initial state.

        Args:
            seed: Random seed (for reproducibility)

        Returns:
            Initial observation
        """
        if seed is not None:
            np.random.seed(seed)

        # Reset car to start position
        self.car.x = self.track.start_pos[0]
        self.car.y = self.track.start_pos[1]
        self.car.angle = self.track.start_angle
        self.car.velocity = 0.0
        self.car.alive = True

        # Reset episode state
        self.current_step = 0
        self.current_checkpoint = 0
        self.total_reward = 0.0
        self.prev_pos = (self.car.x, self.car.y)
        self.position_trail = []
        self.start_time = time.time()  # Track episode start time for finish time calculation
        self.finish_time = None  # Will be set when finish line is crossed

        # Get initial observation
        observation = self.car.get_observation(self.track)

        return observation

    def step(
        self, action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one step in the environment.

        Args:
            action: Action to take (0=LEFT, 1=STRAIGHT, 2=RIGHT)

        Returns:
            observation: Current state observation
            reward: Reward for this step
            terminated: Whether episode ended (crash or completion)
            truncated: Whether episode was truncated (max steps)
            info: Additional information
        """
        # Store previous position for checkpoint detection
        self.prev_pos = (self.car.x, self.car.y)

        # Add to trail
        self.position_trail.append((self.car.x, self.car.y, self.car.angle))
        if len(self.position_trail) > self.max_trail_length:
            self.position_trail.pop(0)

        # Update car physics
        self.car.update(action)
        self.current_step += 1

        # Check collision with track boundaries
        corners = self.car.get_corners()
        on_track = all(self.track.is_point_on_track(x, y) for x, y in corners)

        # Calculate reward
        reward = 0.0
        terminated = False
        truncated = False

        if not on_track:
            # Crashed into wall
            reward = self.reward_crash
            terminated = True
            self.car.kill()
        else:
            # Survived this step
            reward = self.reward_survival

            # CRITICAL: Get current observation for reward shaping
            curr_pos = (self.car.x, self.car.y)
            observation = self.car.get_observation(self.track)
            sensor_readings = observation[:7]  # First 7 values are sensors (now 7 sensors!)

            # 1. WALL PROXIMITY PENALTY - BALANCED BUT CLEAR!
            # Sensor values: 0.0 = far from wall, 1.0 = very close to wall
            # RED sensor (>= 0.8) = DANGER! MUST TURN NOW!
            max_sensor = max(sensor_readings)
            wall_penalty = 0.0

            for sensor_val in sensor_readings:
                if sensor_val >= 0.8:
                    # RED ZONE! CRITICAL DANGER - Strong penalty (reduced 5x)
                    red_penalty = -10.0 * (sensor_val - 0.8) ** 2
                    wall_penalty += red_penalty
                    # Extra penalty for being very red
                    wall_penalty -= 5.0 * sensor_val
                elif sensor_val >= 0.2:
                    # Yellow zone - warning
                    yellow_penalty = -2.0 * ((sensor_val - 0.2) ** 2)
                    wall_penalty += yellow_penalty

            reward += wall_penalty

            # If ANY sensor is red (>= 0.8), add significant penalty
            if max_sensor >= 0.8:
                # Strong signal to turn! (reduced 5x)
                reward -= 20.0 * (max_sensor - 0.8) ** 2

            # If multiple sensors are red at once, worse!
            red_sensor_count = sum(1 for s in sensor_readings if s >= 0.8)
            if red_sensor_count > 1:
                reward -= 10.0 * red_sensor_count

            # 1b. REWARD FOR TURNING AWAY FROM RED SENSORS!
            # 7 Sensors: [left-90, left-60, left-30, center, right-30, right-60, right-90]
            # Actions: 0=LEFT, 1=STRAIGHT, 2=RIGHT
            left_red = max(sensor_readings[0], sensor_readings[1], sensor_readings[2]) >= 0.8
            right_red = max(sensor_readings[4], sensor_readings[5], sensor_readings[6]) >= 0.8
            center_red = sensor_readings[3] >= 0.8

            if left_red and action == 2:  # Left is red, turning RIGHT = GOOD!
                reward += 15.0  # Reward for correct turning (reduced)
            elif right_red and action == 0:  # Right is red, turning LEFT = GOOD!
                reward += 15.0  # Reward for correct turning (reduced)
            elif center_red and action != 1:  # Center is red, turning either way = GOOD!
                reward += 10.0  # Reward for turning away

            # Penalty for turning TOWARDS red sensors (wrong move!)
            if left_red and action == 0:  # Left is red, but turning LEFT = BAD!
                reward -= 20.0  # Clear penalty (reduced)
            elif right_red and action == 2:  # Right is red, but turning RIGHT = BAD!
                reward -= 20.0  # Clear penalty (reduced)

            # 2. CENTER OF TRACK BONUS
            # Reward for having balanced sensor readings (staying centered)
            # With 7 sensors: [L90, L60, L30, C, R30, R60, R90]
            left_sensors = sensor_readings[0] + sensor_readings[1] + sensor_readings[2]
            right_sensors = sensor_readings[4] + sensor_readings[5] + sensor_readings[6]
            balance = 1.0 - abs(left_sensors - right_sensors) / 3.0  # Normalize by 3
            reward += 0.5 * balance

            # 3. Speed bonus (encourage FORWARD movement ONLY, penalize backwards)
            if self.car.velocity > 0:
                # FORWARD movement - GOOD!
                speed_normalized = self.car.velocity / self.car.max_velocity
                if max_sensor < 0.4:  # Only reward speed when SAFE
                    reward += 1.0 * speed_normalized  # Reward for safe forward speed
            else:
                # BACKWARD movement - BAD! Heavy penalty
                reward -= 10.0 * abs(self.car.velocity)  # Strong penalty for going backwards

            # 4. Distance traveled reward (ONLY when moving forward)
            if self.car.velocity > 0:
                distance = np.sqrt(
                    (curr_pos[0] - self.prev_pos[0])**2 +
                    (curr_pos[1] - self.prev_pos[1])**2
                )
                reward += 0.1 * distance

            # 5. Check checkpoint crossing (big bonus or penalty)
            checkpoint_passed = False
            wrong_checkpoint_crossed = False

            # Check if correct checkpoint was crossed
            if self.track.check_checkpoint(
                self.prev_pos, curr_pos, self.current_checkpoint
            ):
                reward += self.reward_checkpoint
                self.current_checkpoint = self.track.get_next_checkpoint_idx(
                    self.current_checkpoint
                )
                checkpoint_passed = True

                # FINISH LINE! If completed all checkpoints (back to 0), WIN!
                if self.current_checkpoint == 0 and checkpoint_passed:
                    # Calculate finish time
                    self.finish_time = time.time() - self.start_time

                    # Base reward for finishing
                    reward += self.reward_finish

                    # TIME BONUS: Apply time-based reward if enabled
                    if self.finish_time_enabled:
                        if self.finish_time_mode == "speed_based":
                            # SPEED-BASED: Semakin cepat = semakin banyak reward
                            # Formula: (max_time - actual_time) × multiplier
                            time_saved = max(0, self.finish_time_max_time - self.finish_time)
                            time_bonus = time_saved * self.finish_time_speed_multiplier
                            reward += time_bonus
                        else:  # target_based
                            # TARGET-BASED: Ada target ideal, ada bonus dan penalty
                            if self.finish_time < self.finish_time_target:
                                # Finished fast! Give big bonus
                                time_bonus = (self.finish_time_target - self.finish_time) * self.finish_time_bonus_multiplier
                                reward += time_bonus
                            else:
                                # Too slow, small penalty
                                time_penalty = (self.finish_time - self.finish_time_target) * self.finish_time_penalty_multiplier
                                reward -= time_penalty

                    terminated = True  # Episode ends successfully

            else:
                # Check if ANY wrong checkpoint was crossed (prevents backward/shortcut)
                for i in range(len(self.track.checkpoints)):
                    if i != self.current_checkpoint:
                        if self.track.check_checkpoint(self.prev_pos, curr_pos, i):
                            # HEAVY PENALTY for crossing wrong checkpoint
                            reward -= 1000.0  # Keras penalty untuk mencegah balik arah
                            wrong_checkpoint_crossed = True
                            break

        # Check if max steps reached
        if self.current_step >= self.max_steps:
            truncated = True

        # Get observation
        observation = self.car.get_observation(self.track)

        # Update total reward
        self.total_reward += reward

        # Build info dict
        # Check if finished (passed all checkpoints)
        num_checkpoints = len(self.track.checkpoints)
        passed_all = self.current_checkpoint == 0 and checkpoint_passed if 'checkpoint_passed' in locals() else False

        info = {
            "step": self.current_step,
            "checkpoint": self.current_checkpoint if not passed_all else num_checkpoints,
            "total_reward": self.total_reward,
            "crashed": not on_track,
            "finished": passed_all,  # NEW: Did agent finish the lap?
            "finish_time": self.finish_time,  # Time to complete lap (None if not finished)
            "position": (self.car.x, self.car.y),
            "velocity": self.car.velocity,
            "angle": self.car.angle,
        }

        return observation, reward, terminated, truncated, info

    def render(self) -> None:
        """Render the current state (if render_mode is set)."""
        if self.render_mode != "human" or not PYGAME_AVAILABLE or self.screen is None:
            return

        # Fill background
        self.screen.fill((34, 34, 34))

        # Render track
        self.track.render(self.screen, show_checkpoints=True)

        # Render position trail with transparency
        self._render_trail()

        # Render car (with sensors)
        self.car.render(self.screen, show_sensors=True)

        # Render info panel
        self._render_info_panel()

        # Update display
        pygame.display.flip()
        self.clock.tick(self.fps)

    def _render_trail(self) -> None:
        """Render transparent trail of previous positions."""
        if not PYGAME_AVAILABLE or self.screen is None:
            return

        num_positions = len(self.position_trail)
        if num_positions < 2:
            return

        # Create temporary surface for transparency
        trail_surface = pygame.Surface((self.window_width, self.window_height))
        trail_surface.set_colorkey((0, 0, 0))  # Black is transparent

        # Draw trail with increasing opacity for recent positions
        for i in range(num_positions - 1):
            x1, y1, angle1 = self.position_trail[i]
            x2, y2, angle2 = self.position_trail[i + 1]

            # Calculate opacity based on position in trail (older = more transparent)
            opacity_factor = i / num_positions
            opacity = int(50 + opacity_factor * 150)  # 50 to 200

            # Draw small car rectangle at this position
            import math
            car_width = self.car.width * 0.8
            car_height = self.car.height * 0.8

            # Calculate car corners at this position
            cos_a = math.cos(angle1)
            sin_a = math.sin(angle1)
            hw, hh = car_width / 2, car_height / 2

            corners = [
                (x1 + cos_a * hw - sin_a * hh, y1 + sin_a * hw + cos_a * hh),
                (x1 + cos_a * hw + sin_a * hh, y1 + sin_a * hw - cos_a * hh),
                (x1 - cos_a * hw + sin_a * hh, y1 - sin_a * hw - cos_a * hh),
                (x1 - cos_a * hw - sin_a * hh, y1 - sin_a * hw + cos_a * hh),
            ]

            # Color changes from blue (old) to cyan (recent)
            color = (100, 150 + int(opacity_factor * 105), 255)
            pygame.draw.polygon(trail_surface, color, corners, 0)

        # Apply transparency and blit to main screen
        trail_surface.set_alpha(180)
        self.screen.blit(trail_surface, (0, 0))

    def _render_info_panel(self) -> None:
        """Render information panel on screen."""
        if not PYGAME_AVAILABLE or self.screen is None:
            return

        info_texts = [
            f"Step: {self.current_step}/{self.max_steps}",
            f"Checkpoint: {self.current_checkpoint}/{len(self.track.checkpoints)}",
            f"Reward: {self.total_reward:.1f}",
            f"Velocity: {self.car.velocity:.2f}",
            f"Alive: {self.car.alive}",
        ]

        y_offset = 10
        for i, text in enumerate(info_texts):
            text_surface = self.font.render(text, True, (255, 255, 255))
            self.screen.blit(text_surface, (10, y_offset + i * 30))

    def close(self) -> None:
        """Clean up resources."""
        if PYGAME_AVAILABLE and pygame.get_init():
            pygame.quit()

    def get_action_space_size(self) -> int:
        """Get number of possible actions."""
        return 3  # LEFT, STRAIGHT, RIGHT

    def get_observation_space_size(self) -> int:
        """Get size of observation vector."""
        return self.num_sensors + 1  # N sensors + 1 velocity

    def __repr__(self) -> str:
        return (
            f"CarEnvironment(track='{self.track.name}', "
            f"step={self.current_step}/{self.max_steps}, "
            f"checkpoint={self.current_checkpoint})"
        )
