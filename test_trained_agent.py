"""
Test trained DQN agent with visualization.
"""

import sys
import argparse
sys.path.insert(0, 'src')

import torch
from environment.simulation import CarEnvironment
from algorithms.dqn.agent import DQNAgent


def test_agent(checkpoint_path: str, num_episodes: int = 5, render: bool = True):
    """
    Test trained agent.

    Args:
        checkpoint_path: Path to saved model
        num_episodes: Number of test episodes
        render: Whether to render (visual)
    """
    print("=" * 60)
    print("🧪 TESTING TRAINED AGENT")
    print("=" * 60)

    # Create environment
    render_mode = "human" if render else None
    env = CarEnvironment(
        track_path="tracks/oval_easy.json",
        render_mode=render_mode,
        max_steps=2000,
    )

    # Create agent
    agent = DQNAgent(
        state_dim=6,
        action_dim=3,
        hidden_dims=[128, 128],  # Match optimized training config
    )

    # Load trained weights
    try:
        agent.load(checkpoint_path)
        print(f"✓ Loaded model from: {checkpoint_path}\n")
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        return

    # Test episodes
    rewards = []
    lengths = []
    checkpoints_passed = []

    for episode in range(num_episodes):
        print(f"\n--- Episode {episode + 1}/{num_episodes} ---")

        state = env.reset()
        episode_reward = 0.0
        episode_length = 0
        done = False

        while not done:
            # Select action (greedy, no exploration)
            action = agent.select_action(state, training=False)

            # Step
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Render
            if render:
                env.render()

            # Update
            state = next_state
            episode_reward += reward
            episode_length += 1

        # Episode stats
        rewards.append(episode_reward)
        lengths.append(episode_length)
        checkpoints_passed.append(info['checkpoint'])

        print(f"Reward: {episode_reward:.2f}")
        print(f"Length: {episode_length} steps")
        print(f"Checkpoints: {info['checkpoint']}/{len(env.track.checkpoints)}")
        print(f"Crashed: {info['crashed']}")

    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    print(f"Episodes: {num_episodes}")
    print(f"Avg Reward: {sum(rewards)/len(rewards):.2f}")
    print(f"Avg Length: {sum(lengths)/len(lengths):.1f} steps")
    print(f"Avg Checkpoints: {sum(checkpoints_passed)/len(checkpoints_passed):.1f}")
    print(f"Best Reward: {max(rewards):.2f}")
    print("=" * 60)

    env.close()


def main():
    parser = argparse.ArgumentParser(description="Test trained DQN agent")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="models/checkpoints/dqn/best_model.pt",
        help="Path to checkpoint",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=5,
        help="Number of test episodes",
    )
    parser.add_argument(
        "--no-render",
        action="store_true",
        help="Disable rendering",
    )
    args = parser.parse_args()

    test_agent(
        checkpoint_path=args.checkpoint,
        num_episodes=args.episodes,
        render=not args.no_render,
    )


if __name__ == "__main__":
    main()
