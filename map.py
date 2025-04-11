import requests
from PIL import Image
import numpy as np
from io import BytesIO
import os
from scipy.ndimage import gaussian_filter, binary_dilation
import random

def get_google_maps_snapshot(lat, lng, api_key, zoom=15, size=(640, 640)):
    """
    Fetch a snapshot from the Google Maps Static API.

    Args:
    - lat (float): Latitude of the center point.
    - lng (float): Longitude of the center point.
    - api_key (str): Google Maps API key.
    - zoom (int): Zoom level for the map.
    - size (tuple): Size of the map image.

    Returns:
    - PIL.Image.Image: The fetched map image.
    """
    base_url = "https://maps.googleapis.com/maps/api/staticmap?"
    scale = 2  # High-resolution scale

    # Style parameter to hide labels and text
    style = [
        "feature:all|element:labels|visibility:off",  # Remove all labels
        "feature:poi|element:labels|visibility:off",  # Remove points of interest labels
        "feature:road|element:labels|visibility:off",  # Remove road labels
    ]

    # Combine styles into a string for the URL
    style_param = "&".join([f"style={s}" for s in style])

    params = {
        "center": f"{lat},{lng}",
        "zoom": zoom,
        "size": f"{size[0]}x{size[1]}",
        "scale": scale,
        "maptype": "default",
        "key": api_key
    }

    # Append the styles to the URL
    response = requests.get(base_url + style_param, params=params)
    if response.status_code == 200:
        return Image.open(BytesIO(response.content))
    else:
        raise Exception(f"Errors: {response.status_code}, {response.text}")

def convert_to_minecraft_palette(image, max_distance=30):
    """
    Convert image colors to a Minecraft-like palette by mapping each pixel to the closest palette color,
    and apply depth algorithm to water areas.

    Args:
    - image (PIL.Image.Image): The input image.
    - max_distance (float): The maximum color distance to consider for mapping.

    Returns:
    - PIL.Image.Image: The transformed image.
    """
    # Minecraft color palette
    # Original color -> Mapped Minecraft color
    palette = {
        'grass': ((211, 248, 226), (108, 152, 47)),    # Light green -> Grass green
        'water': ((144, 218, 238), (64, 63, 252)),     # Light blue -> Water blue
        'light_grey': ((233, 234, 239), (205, 127, 55)),  # Light grey -> Brown
        # Add more mappings if needed
    }

    # Extract the palette colors and their corresponding Minecraft colors
    original_colors = np.array([color[0] for color in palette.values()])
    minecraft_colors = np.array([color[1] for color in palette.values()])

    # Ensure the image is in RGB mode
    if image.mode != 'RGB':
        image = image.convert('RGB')

    # Convert the image to a NumPy array
    image_array = np.array(image)

    # Reshape the image array to a 2D array of pixels
    pixels = image_array.reshape(-1, 3)

    # Compute the distance between each pixel and each color in the palette
    distances = np.linalg.norm(pixels[:, None] - original_colors[None, :], axis=2)

    # Find the minimum distance and the index of the closest palette color for each pixel
    min_distances = np.min(distances, axis=1)
    closest_color_indices = np.argmin(distances, axis=1)

    # Create a mask for pixels within the max_distance
    within_distance = min_distances <= max_distance

    # Map pixels within the threshold to the closest Minecraft color
    new_pixels = pixels.copy()
    new_pixels[within_distance] = minecraft_colors[closest_color_indices[within_distance]]

    # Reshape the new pixels back to the original image shape
    new_image_array = new_pixels.reshape(image_array.shape)

    # Now process the water areas
    new_image_array = process_water_areas(new_image_array, palette)

    # Convert the modified array back to a PIL image
    return Image.fromarray(new_image_array.astype('uint8'))

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
    structure = np.ones((3, 3))
    dilated_water_mask = binary_dilation(water_mask, structure=structure)
    shoreline_mask = water_mask & (~dilated_water_mask | water_mask)

    # Set depth to 0 (shallow) for shoreline pixels
    depth_map[shoreline_mask] = 0

    # For other water pixels, randomly assign deep spots
    # Create a random depth map for water pixels
    random_depth = np.random.rand(*image_array.shape[:2])
    # Smooth the random depth map to create natural-looking deep spots
    smooth_depth = gaussian_filter(random_depth, sigma=5)

    # Normalize and threshold to create deep spots
    deep_threshold = 0.6  # Adjust this value to control the amount of deep water
    deep_spots = (smooth_depth > deep_threshold) & water_mask & (~shoreline_mask)
    depth_map[deep_spots] = 1  # Set depth to 1 (deep) for deep spots

    # Apply checker pattern to deep water
    checkerboard = (np.indices(depth_map.shape).sum(axis=0) % 2 == 0)
    deep_water_mask = (depth_map == 1)
    image_array[deep_water_mask & checkerboard] = minecraft_water_color
    image_array[deep_water_mask & ~checkerboard] = deep_water_color
    print(f"Number of water pixels: {np.sum(water_mask)}")
    print(f"Number of shoreline pixels: {np.sum(shoreline_mask)}")
    print(f"Number of deep water pixels after thresholding: {np.sum(deep_spots)}")
    print(f"Number of deep water pixels in depth map: {np.sum(deep_water_mask)}")
    
    return image_array

def add_fake_elevation(image_array, elevation_levels=5):
    """
    Add fake elevation to grass areas in an image and apply vertical scan shading.
    Returns the modified image array and elevation map.
    """
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

def generate_minecraft_map(lat, lng, api_key, output_dir="./"):
    """
    Generate a Minecraft-style map image with fake elevation, trees, processed water areas,
    and overlay it onto a scaled outline map.
    
    Args:
    - lat (float): Latitude of the center point.
    - lng (float): Longitude of the center point.
    - api_key (str): Google Maps API key.
    - output_dir (str): Directory to save output images.
    
    Returns:
    - PIL.Image.Image: The final image with overlay.
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Get the snapshot from Google Maps
    try:
        original_map = get_google_maps_snapshot(lat, lng, api_key)
    
        # Save the original map
        original_map_path = os.path.join(output_dir, "original_map.png")
        original_map.save(original_map_path)
    
        # Resize to 128x128 and save it
        resized_map = original_map.resize((128, 128))
        resized_map_path = os.path.join(output_dir, "resized_map.png")
        resized_map.save(resized_map_path)
    
        # Convert to Minecraft palette and save it
        minecraft_map = convert_to_minecraft_palette(resized_map, max_distance=30)
        minecraft_map_path = os.path.join(output_dir, "minecraft_map.png")
        minecraft_map.save(minecraft_map_path)
    
        # Convert the Minecraft map to a NumPy array for further processing
        minecraft_map_array = np.array(minecraft_map)
    
        # Add fake elevation
        elevated_image_array, elevation_map = add_fake_elevation(minecraft_map_array)
    
        # Add trees
        image_with_trees_array = add_trees(elevated_image_array, elevation_map, tree_count=30)
    
        # Scale up the image
        scale_factor = 4  # Adjust the scale factor as needed
        scaled_image = scale_up_image(image_with_trees_array, scale_factor=scale_factor)
    
        # Save the scaled image
        scaled_image_path = os.path.join(output_dir, "map.png")
        scaled_image.save(scaled_image_path)
        print(f"Scaled image saved to: {scaled_image_path}")
    
        # Load and scale the outline map
        outline_map = Image.open('./Map.png')
    
        # Scale the outline map
        original_size = outline_map.size
        new_size = (original_size[0] * 4, original_size[1] * scale_factor)
        scaled_outline_map = outline_map.resize(new_size, resample=Image.NEAREST)
    
        # Save the scaled outline map
        scaled_outline_map_path = os.path.join(output_dir, "scaled_outline_map.png")
        scaled_outline_map.save(scaled_outline_map_path)
        print(f"Scaled outline map saved to: {scaled_outline_map_path}")
    
        # Overlay the generated map onto the scaled outline map
        outline_width, outline_height = scaled_outline_map.size
        map_width, map_height = scaled_image.size
    
        # Calculate position to center the map on the outline
        x = (outline_width - map_width) // 2
        y = (outline_height - map_height) // 2
    
        # Ensure the generated map has an alpha channel
        if scaled_image.mode != 'RGBA':
            scaled_image = scaled_image.convert('RGBA')
    
        # Ensure the outline map is in RGB mode
        if scaled_outline_map.mode != 'RGBA':
            scaled_outline_map = scaled_outline_map.convert('RGBA')
    
        # Create a copy of the outline map to paste onto
        final_image = scaled_outline_map.copy()
    
        # Paste the generated map onto the outline map, centered
        final_image.paste(scaled_image, (x, y), scaled_image)
    
        # Save the final image
        final_image_path = os.path.join(output_dir, "final_map_overlay.png")
        final_image.save(final_image_path)
        print(f"Final image with overlay saved to: {final_image_path}")
    
        return final_image
    
    except Exception as e:
        print(f"Error: {e}")
        return None


# Example usage
if __name__ == "__main__":
    # Coordinates (latitude, longitude)
    # latitude = 42.145242  # Replace with your latitude 
    # longitude = -88.001543  # Replace with your longitude
    latitude = 42.145242  # Replace with your latitude
    longitude = -88.001543  # Replace with your longitude
    #42.250595, -88.050519
    api_key = "KEY"  # Replace with your API key

    final_image = generate_minecraft_map(latitude, longitude, api_key)
    if final_image:
        final_image.show()  # Show the final image
