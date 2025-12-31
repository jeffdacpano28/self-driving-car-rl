"""
Configuration loader utilities.
Provides centralized config loading for all training scripts.
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional


def load_environment_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load environment configuration from YAML file.

    Args:
        config_path: Path to config file (default: config/environment.yaml)

    Returns:
        Dictionary with environment configuration
    """
    if config_path is None:
        # Default to config/environment.yaml relative to project root
        project_root = Path(__file__).parent.parent.parent
        config_path = project_root / "config" / "environment.yaml"

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    return config


def get_state_dim_from_config(config: Dict[str, Any]) -> int:
    """
    Calculate state dimension from environment config.

    State = num_sensors + 1 (velocity)

    Args:
        config: Environment configuration dict

    Returns:
        State dimension (int)
    """
    num_sensors = config['sensors']['num_sensors']
    return num_sensors + 1  # sensors + velocity


def get_action_dim_from_config(config: Dict[str, Any]) -> int:
    """
    Get action dimension from environment config.

    Args:
        config: Environment configuration dict

    Returns:
        Action dimension (9 for discrete control: 3 steering × 3 speed)
    """
    action_config = config.get('actions', {})
    return action_config.get('num_actions', 9)  # Default to 9


def get_car_params_from_config(config: Dict[str, Any]) -> Dict[str, float]:
    """
    Extract car physics parameters from config.

    Args:
        config: Environment configuration dict

    Returns:
        Dict with car physics parameters
    """
    car_config = config['car']
    return {
        'car_width': float(car_config['width']),
        'car_height': float(car_config['height']),
        'car_max_velocity': float(car_config['max_velocity']),
        'car_min_velocity': float(car_config['min_velocity']),
        'car_acceleration': float(car_config['acceleration']),
        'car_friction': float(car_config['friction']),
        'car_turn_rate': float(car_config['turn_rate']),
        'car_angular_damping': float(car_config.get('angular_damping', 0.85)),
    }


def get_sensor_params_from_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract sensor parameters from config.

    Args:
        config: Environment configuration dict

    Returns:
        Dict with sensor parameters
    """
    sensor_config = config['sensors']
    return {
        'num_sensors': sensor_config['num_sensors'],
        'angles': sensor_config['angles'],
        'max_range': float(sensor_config['max_range']),
    }


def get_reward_params_from_config(config: Dict[str, Any]) -> Dict[str, float]:
    """
    Extract reward parameters from config.

    Args:
        config: Environment configuration dict

    Returns:
        Dict with reward parameters
    """
    reward_config = config['rewards']
    return {
        'reward_survival': float(reward_config['survival']),
        'reward_checkpoint': float(reward_config['checkpoint']),
        'reward_crash': float(reward_config['crash']),
        'reward_finish': float(reward_config.get('finish', 1000.0)),
    }


def get_finish_time_params_from_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract finish time bonus parameters from config.

    Args:
        config: Environment configuration dict

    Returns:
        Dict with finish time parameters
    """
    finish_time_config = config.get('finish_time', {})
    return {
        'finish_time_enabled': finish_time_config.get('enabled', True),
        'finish_time_mode': finish_time_config.get('mode', 'speed_based'),
        # Speed-based mode params
        'finish_time_max_time': float(finish_time_config.get('max_time', 60.0)),
        'finish_time_speed_multiplier': float(finish_time_config.get('speed_multiplier', 50.0)),
        # Target-based mode params
        'finish_time_target': float(finish_time_config.get('target_time', 30.0)),
        'finish_time_bonus_multiplier': float(finish_time_config.get('bonus_multiplier', 50.0)),
        'finish_time_penalty_multiplier': float(finish_time_config.get('penalty_multiplier', 10.0)),
    }


def get_episode_params_from_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract episode parameters from config.

    Args:
        config: Environment configuration dict

    Returns:
        Dict with episode parameters
    """
    episode_config = config['episode']
    return {
        'max_steps': episode_config['max_steps'],
    }
