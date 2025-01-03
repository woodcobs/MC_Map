import math
import pygame
import sys

pygame.init()

# Set up the display
WIDTH, HEIGHT = 720, 720
screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.FULLSCREEN)
pygame.display.set_caption("Simple Map with Movable Cursor")

# Load and scale the map image
map_image = pygame.image.load("./map.png")
map_image = pygame.transform.scale(map_image, (WIDTH, HEIGHT))  # Ensure the map is resized to fit the canvas

# Updated cursor pattern
cursor_pattern = [
    "xxBxx",
    "xBGBx",
    "BGWGB",
    "BWWWB",
    "BWWWB",
    "BGWGB",
    "xBBBx"
]

# Define colors for cursor pattern
colors = {
    'B': (0, 0, 0),         # Black
    'G': (200, 200, 200),   # Light Grey
    'W': (255, 255, 255),   # White
    'x': None                # No pixel
}

# Draw the cursor pattern onto a surface
BLOCK_SIZE = 5
cursor_width = len(cursor_pattern[0])
cursor_height = len(cursor_pattern)

cursor_surface = pygame.Surface((cursor_width * BLOCK_SIZE, cursor_height * BLOCK_SIZE), pygame.SRCALPHA)
cursor_surface.fill((0, 0, 0, 0))  # Transparent background

for row_i, row in enumerate(cursor_pattern):
    for col_i, char in enumerate(row):
        color = colors[char]
        if color is not None:
            rect = pygame.Rect(
                col_i * BLOCK_SIZE,
                row_i * BLOCK_SIZE,
                BLOCK_SIZE,
                BLOCK_SIZE
            )
            pygame.draw.rect(cursor_surface, color, rect)

# Cursor initial position and angle
cursor_pos = [WIDTH // 2, HEIGHT // 2]
cursor_angle = 0.0

# Movement parameters
move_speed = 2
rotate_speed = 5

# Key states for continuous movement
keys_held = {
    pygame.K_UP: False,
    pygame.K_DOWN: False,
    pygame.K_LEFT: False,
    pygame.K_RIGHT: False
}

running = True
clock = pygame.time.Clock()

while running:
    dt = clock.tick(60)  # Frame rate
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
            break

        if event.type == pygame.KEYDOWN:
            if event.key in keys_held:
                keys_held[event.key] = True
            # **Added: Pressing 'q' quits the game**
            if event.key == pygame.K_q:
                running = False
                break

        if event.type == pygame.KEYUP:
            if event.key in keys_held:
                keys_held[event.key] = False

    # Handle rotation: 
    # LEFT key rotates counterclockwise (increasing angle),
    # RIGHT key rotates clockwise (decreasing angle).
    if keys_held[pygame.K_LEFT]:
        cursor_angle += rotate_speed
    if keys_held[pygame.K_RIGHT]:
        cursor_angle -= rotate_speed

    # Convert angle to radians for movement calculations
    angle_radians = math.radians(cursor_angle)
    # Direction vector based on angle
    # At angle=0 (up), we want (0, -1)
    # At angle=90 (left), we want (-1, 0), etc.
    forward_dx = -math.sin(angle_radians)
    forward_dy = -math.cos(angle_radians)

    # Movement forward/backward
    if keys_held[pygame.K_UP]:
        cursor_pos[0] += forward_dx * move_speed
        cursor_pos[1] += forward_dy * move_speed
    if keys_held[pygame.K_DOWN]:
        cursor_pos[0] -= forward_dx * move_speed
        cursor_pos[1] -= forward_dy * move_speed

    # Clamp the cursor within the window
    cursor_pos[0] = max(0, min(WIDTH, cursor_pos[0]))
    cursor_pos[1] = max(0, min(HEIGHT, cursor_pos[1]))

    # Draw the map
    screen.blit(map_image, (0, 0))

    # Rotate and draw the cursor
    rotated_cursor = pygame.transform.rotate(cursor_surface, cursor_angle)
    rc_rect = rotated_cursor.get_rect(center=(cursor_pos[0], cursor_pos[1]))
    screen.blit(rotated_cursor, rc_rect)

    pygame.display.flip()

pygame.quit()
sys.exit()
