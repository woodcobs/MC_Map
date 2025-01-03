from PIL import Image
import numpy as np
from scipy.ndimage import gaussian_filter
import os
import random

def add_fake_elevation(image_path, elevation_levels=5):
    """
    Add fake elevation to grass areas in an image and apply vertical scan shading.
    Returns the modified image array and elevation map.
    """
    # Load the image
    image = Image.open(image_path)
    image_array = np.array(image)

    # Define grass color
    grass_color = np.array([108, 152, 47])  # Base grass color

    # Create a mask of grass pixels
    grass_mask = np.all(image_array == grass_color, axis=-1)

    # Generate random noise over grass pixels
    noise = np.zeros(image_array.shape[:2])
    noise[grass_mask] = np.random.rand(np.count_nonzero(grass_mask))

    # Apply Gaussian filter to smooth the noise
    smooth_noise = gaussian_filter(noise, sigma=5)

    # Normalize the elevation map over grass pixels
    elevation = smooth_noise[grass_mask]
    min_elev, max_elev = elevation.min(), elevation.max()
    elevation_normalized = (elevation - min_elev) / (max_elev - min_elev)  # Normalize to 0-1

    # Quantize elevation into levels
    N_levels = elevation_levels  # Number of levels
    elevation_levels_array = np.floor(elevation_normalized * N_levels).astype(int)
    elevation_levels_array = np.clip(elevation_levels_array, 0, N_levels - 1)  # Ensure levels are between 0 and N-1

    # Create an elevation map with default -1 for non-grass pixels
    elevation_map = np.full(image_array.shape[:2], -1, dtype=int)
    elevation_map[grass_mask] = elevation_levels_array

    # Define shading colors
    base_grass_color = grass_color
    light_color = np.array([154, 205, 50])  # Light green
    dark_color = np.array([85, 107, 47])    # Dark green

    # Copy the original image array to modify
    new_image_array = image_array.copy()

    # Perform vertical scan
    for j in range(image_array.shape[1]):  # For each column
        for i in range(image_array.shape[0]):  # For each row from top to bottom
            if elevation_map[i, j] != -1:  # If it's a grass pixel
                current_level = elevation_map[i, j]
                if i > 0 and elevation_map[i - 1, j] != -1:
                    prev_level = elevation_map[i - 1, j]
                    if current_level > prev_level:
                        # Step up, make it light
                        new_image_array[i, j] = light_color
                    elif current_level < prev_level:
                        # Step down, make it dark
                        new_image_array[i, j] = dark_color
                    else:
                        # No change, base grass color
                        new_image_array[i, j] = base_grass_color
                else:
                    # First row or no grass pixel above
                    new_image_array[i, j] = base_grass_color

    return new_image_array, elevation_map

def add_trees(image_array, elevation_map, tree_count=30):
    """
    Add trees to the image array at random locations away from elevation changes.
    
    Args:
    - image_array (numpy.ndarray): The image array to modify.
    - elevation_map (numpy.ndarray): The elevation map corresponding to the image.
    - tree_count (int): Number of trees to attempt to place.
    
    Returns:
    - new_image_array (numpy.ndarray): The image array with trees added.
    """
    # Define grass color
    grass_color = np.array([108, 152, 47])  # Base grass color
    base_grass_color = grass_color
    light_color = np.array([154, 205, 50])  # Light green
    dark_color = np.array([85, 107, 47])    # Dark green

    # Define tree colors
    tree_color_outer = np.array([0, 122, 1])
    tree_color_inner1 = np.array([0, 122, 1])
    tree_color_inner2 = np.array([1, 85, 0])

    # Dimensions
    height, width = image_array.shape[:2]
    tree_size = 5  # Tree is 5x5 pixels

    # Create a copy of the image array to modify
    new_image_array = image_array.copy()

    # Identify suitable locations for trees
    suitable_locations = []

    # We need to avoid the edges to fit a 5x5 tree
    margin = tree_size // 2

    for attempt in range(tree_count * 10):  # Allow multiple attempts to find suitable locations
        # Randomly select a center position
        i = random.randint(margin, height - margin - 1)
        j = random.randint(margin, width - margin - 1)

        # Check if the area is suitable for placing a tree
        area_elevation = elevation_map[i - margin:i + margin + 1, j - margin:j + margin + 1]
        area_pixels = new_image_array[i - margin:i + margin + 1, j - margin:j + margin + 1]

        # Check if all pixels in the area are grass and have the same elevation
        if np.all(area_elevation != -1) and np.all(area_elevation == area_elevation[margin, margin]):
            # Check if the area does not already have a tree (we can check if it's grass color)
            area_colors = area_pixels.reshape(-1, 3)
            if np.all(
                np.any(
                    (area_colors == base_grass_color).all(axis=1) |
                    (area_colors == light_color).all(axis=1) |
                    (area_colors == dark_color).all(axis=1),
                    axis=0
                )
            ):
                suitable_locations.append((i, j))
                if len(suitable_locations) >= tree_count:
                    break  # We have enough locations

    # Now, draw the trees at the suitable locations
    for (i_center, j_center) in suitable_locations:
        # Define the tree pattern
        # Create a 5x5 array representing the tree
        tree_pattern = np.full((tree_size, tree_size, 3), tree_color_outer, dtype=np.uint8)

        # Remove the corners (set them to base grass color)
        tree_pattern[0, 0] = base_grass_color
        tree_pattern[0, -1] = base_grass_color
        tree_pattern[-1, 0] = base_grass_color
        tree_pattern[-1, -1] = base_grass_color

        # Invert the checker pattern in the middle 3x3 area
        for x in range(1, 4):
            for y in range(1, 4):
                if (x + y) % 2 == 0:
                    tree_pattern[x, y] = tree_color_inner2  # Swapped colors
                else:
                    tree_pattern[x, y] = tree_color_inner1

        # Place the tree pattern onto the image
        i_start = i_center - margin
        i_end = i_center + margin + 1
        j_start = j_center - margin
        j_end = j_center + margin + 1

        new_image_array[i_start:i_end, j_start:j_end] = tree_pattern

    return new_image_array

def process_water_areas(image_array, palette):
    """
    Process water areas to assign depths and apply patterns.

    Args:
    - image_array (numpy.ndarray): The image array after initial color mapping.
    - palette (dict): The color palette used for mapping.

    Returns:
    - numpy.ndarray: The modified image array with water depths applied.
    """
    # Extract Minecraft water color
    minecraft_water_color = np.array(palette['water'][1])

    # Define colors for deep water
    deep_water_color = np.array([0, 0, 139])  # Darker blue for deep water

    # Create a mask for water pixels
    water_mask = np.all(image_array == minecraft_water_color, axis=-1)

    # Create a depth map for water pixels (default 0 for shallow)
    depth_map = np.zeros(image_array.shape[:2], dtype=int)

    # Ensure shoreline is shallow
    # First, identify shoreline pixels (water pixels adjacent to non-water pixels)
    from scipy.ndimage import binary_dilation

    # Dilate the water mask to find adjacent areas
    dilated_water_mask = binary_dilation(water_mask, structure=np.ones((3, 3)))

    # Shoreline pixels are water pixels adjacent to non-water pixels
    shoreline_mask = water_mask & (~dilated_water_mask | water_mask)

    # Set depth to 0 (shallow) for shoreline pixels
    depth_map[shoreline_mask] = 0

    # For other water pixels, randomly assign deep spots
    # Create a random depth map for water pixels
    random_depth = np.random.rand(*image_array.shape[:2])
    # Smooth the random depth map to create natural-looking deep spots
    from scipy.ndimage import gaussian_filter
    smooth_depth = gaussian_filter(random_depth, sigma=5)

    # Normalize and threshold to create deep spots
    deep_threshold = 0.6  # Adjust this value to control the amount of deep water
    deep_spots = (smooth_depth > deep_threshold) & water_mask & (~shoreline_mask)
    depth_map[deep_spots] = 1  # Set depth to 1 (deep) for deep spots

    # Apply checker pattern to deep water
    for i in range(image_array.shape[0]):
        for j in range(image_array.shape[1]):
            if depth_map[i, j] == 1:
                # Apply checker pattern
                if (i + j) % 2 == 0:
                    image_array[i, j] = minecraft_water_color
                else:
                    image_array[i, j] = deep_water_color
            elif depth_map[i, j] == 0 and water_mask[i, j]:
                # Shallow water, keep original water color
                image_array[i, j] = minecraft_water_color

    return image_array


def scale_up_image(image_array, scale_factor=4):
    """
    Scale up the image while keeping it pixelated.

    Args:
    - image_array (numpy.ndarray): The image array to scale up.
    - scale_factor (int): The factor by which to scale the image dimensions.

    Returns:
    - scaled_image (PIL.Image.Image): The scaled-up image.
    """
    image = Image.fromarray(image_array)
    new_size = (image.width * scale_factor, image.height * scale_factor)
    scaled_image = image.resize(new_size, resample=Image.NEAREST)
    return scaled_image

# Example usage
if __name__ == "__main__":
    input_path = "./output/minecraft_map.png"  # Path to the input image
    output_path = "./output/minecraft_map_with_elevation_and_trees_scaled.png"  # Output file path

    # First, add fake elevation
    new_image_array, elevation_map = add_fake_elevation(input_path)

    # Then, add trees
    new_image_array_with_trees = add_trees(new_image_array, elevation_map, tree_count=30)

    # Scale up the image
    scale_factor = 4  # Adjust the scale factor as needed
    scaled_image = scale_up_image(new_image_array_with_trees, scale_factor=scale_factor)

    # Save the final image
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    scaled_image.save(output_path)
    print(f"Fake elevation and trees added and saved to: {output_path}")
