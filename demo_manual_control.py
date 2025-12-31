"""
Manual keyboard control demo for testing the environment.

Controls:
- LEFT ARROW: Turn left
- RIGHT ARROW: Turn right
- UP ARROW: Go straight (accelerate)
- R: Reset episode
- Q or ESC: Quit
"""

import sys
sys.path.insert(0, 'src')

import pygame
from environment.simulation import CarEnvironment
from environment.car import Car


def main():
    print("=" * 60)
    print("MANUAL CONTROL DEMO")
    print("=" * 60)
    print("\nControls:")
    print("  LEFT ARROW  - Turn left")
    print("  RIGHT ARROW - Turn right")
    print("  UP ARROW    - Go straight (accelerate)")
    print("  R           - Reset episode")
    print("  Q or ESC    - Quit")
    print("\n" + "=" * 60)

    # Create environment with rendering
    env = CarEnvironment(
        track_path="tracks/oval_easy.json",
        render_mode="human",
        max_steps=2000,
    )

    # Reset environment
    observation = env.reset()
    print(f"\n✓ Environment initialized")
    print(f"  Track: {env.track.name}")
    print(f"  Observation size: {len(observation)}")
    print(f"  Start position: {env.track.start_pos}")
    print(f"\n🎮 Use arrow keys to control the car!\n")

    running = True
    episode_active = True  # Track if episode is ongoing
    action = Car.STRAIGHT  # Default action

    while running:
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.KEYDOWN:
                if event.key in [pygame.K_q, pygame.K_ESCAPE]:
                    running = False
                elif event.key == pygame.K_r:
                    # Reset episode
                    observation = env.reset()
                    episode_active = True
                    print("🔄 Episode reset")

        # Only step if episode is active
        if episode_active:
            # Get keyboard state
            keys = pygame.key.get_pressed()

            # Determine action based on keys
            if keys[pygame.K_LEFT]:
                action = Car.TURN_LEFT
            elif keys[pygame.K_RIGHT]:
                action = Car.TURN_RIGHT
            elif keys[pygame.K_UP]:
                action = Car.STRAIGHT
            else:
                # No key pressed, maintain straight
                action = Car.STRAIGHT

            # Step environment
            observation, reward, terminated, truncated, info = env.step(action)

            # Check if episode ended
            if terminated:
                episode_active = False
                if info["crashed"]:
                    print(f"💥 Crashed at step {info['step']}!")
                    print(f"   Final reward: {info['total_reward']:.1f}")
                    print(f"   Checkpoints: {info['checkpoint']}/{len(env.track.checkpoints)}")
                    print("   Press R to reset\n")

            if truncated:
                episode_active = False
                print(f"⏱️  Max steps reached!")
                print(f"   Final reward: {info['total_reward']:.1f}")
                print(f"   Checkpoints: {info['checkpoint']}/{len(env.track.checkpoints)}")
                print("   Press R to reset\n")

            # Print checkpoint progress (only if episode active)
            if reward > env.reward_survival:  # Checkpoint passed
                print(f"✅ Checkpoint {info['checkpoint']-1} passed! (+{env.reward_checkpoint})")

        # Always render (even when paused)
        env.render()

    # Cleanup
    env.close()
    print("\n👋 Demo ended. Goodbye!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Demo interrupted. Goodbye!")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
