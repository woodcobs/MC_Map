import math
import pygame
import sys

pygame.init()

# Set up the display
WIDTH, HEIGHT = 720, 720
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Pixelated Reveal Map with Movable Cursor")

# Load and scale the map image
map_image = pygame.image.load("./original_map.png")
map_image = pygame.transform.scale(map_image, (WIDTH, HEIGHT))

# Create a fog surface to cover the entire map
fog = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
fog.fill((210,180,140,255))  # Tan color

# Reveal settings
REVEAL_RADIUS = 80
BLOCK_SIZE = 5  # Larger block size for more pixelated reveal

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

colors = {
    'B': (0,0,0),
    'G': (200,200,200),
    'W': (255,255,255),
    'x': None  # no pixel drawn
}

# Draw the cursor pattern onto a surface
cursor_width = len(cursor_pattern[0])/2
cursor_height = len(cursor_pattern)/2

cursor_surface = pygame.Surface((cursor_width*BLOCK_SIZE, cursor_height*BLOCK_SIZE), pygame.SRCALPHA)
cursor_surface.fill((0,0,0,0))  # transparent background

for row_i, row in enumerate(cursor_pattern):
    for col_i, char in enumerate(row):
        color = colors[char]
        if color is not None:
            rect = pygame.Rect(
                col_i*BLOCK_SIZE,
                row_i*BLOCK_SIZE,
                BLOCK_SIZE,
                BLOCK_SIZE
            )
            pygame.draw.rect(cursor_surface, color, rect)

# Cursor initial position and angle
cursor_pos = [WIDTH//2, HEIGHT//2]

# Angle definition:
# 0 degrees = Up
# Angle increases counterclockwise:
#  90 deg = Left, 180 deg = Down, 270 deg = Right
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

def reveal_around_point(cx, cy):
    # Reveal a pixelated area around the given point (cx, cy)
    start_x = max(int(cx - REVEAL_RADIUS), 0)
    end_x = min(int(cx + REVEAL_RADIUS), WIDTH - 1)
    start_y = max(int(cy - REVEAL_RADIUS), 0)
    end_y = min(int(cy + REVEAL_RADIUS), HEIGHT - 1)

    fog.lock()
    for y in range(start_y, end_y, BLOCK_SIZE):
        for x in range(start_x, end_x, BLOCK_SIZE):
            # center of this block
            block_cx = x + BLOCK_SIZE // 2
            block_cy = y + BLOCK_SIZE // 2
            dist_sq = (block_cx - cx)**2 + (block_cy - cy)**2
            if dist_sq <= REVEAL_RADIUS**2:
                # Erase BLOCK_SIZE area
                for yy in range(y, min(y+BLOCK_SIZE, HEIGHT)):
                    for xx in range(x, min(x+BLOCK_SIZE, WIDTH)):
                        fog.set_at((xx, yy), (0,0,0,0))
    fog.unlock()

running = True
clock = pygame.time.Clock()

# Initially reveal at the starting position
reveal_around_point(cursor_pos[0], cursor_pos[1])

while running:
    dt = clock.tick(60)  # Frame rate
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
            break

        if event.type == pygame.KEYDOWN:
            if event.key in keys_held:
                keys_held[event.key] = True
        if event.type == pygame.KEYUP:
            if event.key in keys_held:
                keys_held[event.key] = False

        if event.type == pygame.MOUSEBUTTONDOWN:
            # Mouse click reveal (optional extra reveal)
            mouse_x, mouse_y = event.pos
            reveal_around_point(mouse_x, mouse_y)

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

    # Clamp the cursor within the window (optional)
    cursor_pos[0] = max(0, min(WIDTH, cursor_pos[0]))
    cursor_pos[1] = max(0, min(HEIGHT, cursor_pos[1]))

    # Always reveal around cursor as it moves
    reveal_around_point(cursor_pos[0], cursor_pos[1])

    # Draw the map
    screen.blit(map_image, (0,0))
    # Draw the fog
    screen.blit(fog, (0,0))

    # Rotate and draw the cursor
    rotated_cursor = pygame.transform.rotate(cursor_surface, cursor_angle)
    rc_rect = rotated_cursor.get_rect(center=(cursor_pos[0], cursor_pos[1]))
    screen.blit(rotated_cursor, rc_rect)

    pygame.display.flip()

pygame.quit()
sys.exit()
