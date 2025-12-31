"""
View and explore track layouts.
Shows the track, checkpoints, and start position clearly.
"""

import sys
sys.path.insert(0, 'src')

import pygame
from environment.track import Track
import argparse


def view_track(track_path: str):
    """
    Display track layout in a window.

    Args:
        track_path: Path to track JSON file
    """
    # Initialize pygame
    pygame.init()

    # Load track
    track = Track.load(track_path)
    print(f"🏁 Viewing Track: {track.name}")
    print(f"   Checkpoints: {len(track.checkpoints)}")
    print(f"   Start Position: {track.start_pos}")
    print(f"\n🎮 Controls:")
    print("   - ESC or Q: Quit")
    print("   - Click anywhere to see coordinates")
    print()

    # Create window
    width, height = 1000, 700
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption(f"Track Viewer - {track.name}")
    clock = pygame.time.Clock()

    # Font for text
    font = pygame.font.Font(None, 24)
    small_font = pygame.font.Font(None, 18)

    running = True
    mouse_pos = (0, 0)

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
            elif event.type == pygame.MOUSEMOTION:
                mouse_pos = event.pos
            elif event.type == pygame.MOUSEBUTTONDOWN:
                x, y = event.pos
                on_track = track.is_point_on_track(x, y)
                print(f"Clicked: ({x}, {y}) - On track: {on_track}")

        # Clear screen
        screen.fill((20, 20, 20))

        # Draw track
        track.render(screen, show_checkpoints=True)

        # Draw start position with larger marker
        start_x, start_y = track.start_pos
        # Draw circle
        pygame.draw.circle(screen, (0, 255, 0), (int(start_x), int(start_y)), 15, 3)
        # Draw cross
        pygame.draw.line(screen, (0, 255, 0),
                        (start_x - 10, start_y), (start_x + 10, start_y), 2)
        pygame.draw.line(screen, (0, 255, 0),
                        (start_x, start_y - 10), (start_x, start_y + 10), 2)

        # Draw start text
        start_text = small_font.render("START", True, (0, 255, 0))
        screen.blit(start_text, (start_x + 20, start_y - 10))

        # Draw checkpoint numbers
        for i, checkpoint in enumerate(track.checkpoints):
            # Get checkpoint center (checkpoint is a Line object)
            start = checkpoint.start
            end = checkpoint.end
            center_x = (start[0] + end[0]) / 2
            center_y = (start[1] + end[1]) / 2

            # Draw checkpoint number
            cp_text = small_font.render(f"CP{i+1}", True, (255, 255, 0))
            screen.blit(cp_text, (center_x - 15, center_y - 10))

        # Draw info panel
        info_y = 10
        info_texts = [
            f"Track: {track.name}",
            f"Checkpoints: {len(track.checkpoints)}",
            f"Start: ({int(start_x)}, {int(start_y)})",
            f"Mouse: ({mouse_pos[0]}, {mouse_pos[1]})",
        ]

        for i, text in enumerate(info_texts):
            text_surface = font.render(text, True, (255, 255, 255))
            # Background for readability
            bg_rect = text_surface.get_rect()
            bg_rect.topleft = (10, info_y + i * 30)
            bg_rect.inflate_ip(10, 5)
            pygame.draw.rect(screen, (40, 40, 40), bg_rect)
            # Text
            screen.blit(text_surface, (10, info_y + i * 30))

        # Draw legend
        legend_y = height - 120
        legend_texts = [
            ("Green Circle: Start Position", (0, 255, 0)),
            ("Yellow Lines: Checkpoints", (255, 255, 0)),
            ("Gray: Track Surface", (100, 100, 100)),
            ("Black: Out of Bounds", (50, 50, 50)),
        ]

        for i, (text, color) in enumerate(legend_texts):
            # Color box
            pygame.draw.rect(screen, color, (10, legend_y + i * 25, 20, 20))
            # Text
            text_surface = small_font.render(text, True, (255, 255, 255))
            screen.blit(text_surface, (35, legend_y + i * 25))

        # Update display
        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
    print("\n👋 Track viewer closed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="View track layout")
    parser.add_argument(
        "--track",
        type=str,
        default="tracks/f1_grand_circuit.json",
        help="Path to track file",
    )
    args = parser.parse_args()

    view_track(args.track)
