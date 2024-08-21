import os
import cv2
import numpy as np
from PIL import Image
# from mapping_utils import deformation3d, generate_zstore
from au_attack import get_max_node

# Constants
INPUT_DIR = '/home/tpei0009/RHDE/CelebA_HQ_face_gender_dataset_test'
OUTPUT_DIR = '/home/tpei0009/RHDE/celebahq_gender'
STICKER_PATHS = ["/ibm/gpfs/home/tpei0009/sticker_test.png", "/ibm/gpfs/home/tpei0009/STSTNet/sun_rm.png", "/ibm/gpfs/home/tpei0009/STSTNet/heart_sticker_rm.png", "/home/tpei0009/MMNet/star_cartoon.png"]
STICKER_NAMES = ['paul', 'sun', 'heart', 'star']
SHAPE_PREDICTOR_PATH = '../STSTNet/shape5_predictor_68_face_landmarks.dat'
MAGNIFICATION_VALUES = [5.0, 1.0, 1.0, 5.0]
IMAGE_EXTENSIONS = [".jpg", ".png", ".jpeg"]

def get_all_image_paths(directory, extensions=IMAGE_EXTENSIONS):
    """Get all image paths in a directory (including subdirectories)."""
    image_paths = []
    for root, _, files in os.walk(directory):
        for file in files:
            if any(file.lower().endswith(ext) for ext in extensions):
                image_paths.append(os.path.join(root, file))
    return image_paths

def create_output_path(input_path, input_dir, output_dir, sticker_name):
    """Create the output directory structure."""
    relative_path = os.path.relpath(input_path, input_dir)
    output_path = os.path.join(output_dir, sticker_name, relative_path)
    output_dir_path = os.path.dirname(output_path)
    os.makedirs(output_dir_path, exist_ok=True)
    return output_path

def make_basemap(width, height, sticker, x, y):
    """Create a basemap for the sticker."""
    layer = Image.new('RGBA', (width, height), (255, 255, 255, 0))  # white and transparent
    layer.paste(sticker, (x, y))
    base = np.array(layer)
    alpha_matrix = base[:, :, 3]
    basemap = np.where(alpha_matrix != 0, 1, 0)
    return base, basemap

def make_stick2(backimg, sticker, x, y, factor=1):
    """Overlay the sticker on the background image."""
    backimg = np.array(backimg)
    r, g, b = cv2.split(backimg)
    background = cv2.merge([b, g, r])
    base, _ = make_basemap(background.shape[1], background.shape[0], sticker, x=x, y=y)

    r, g, b, a = cv2.split(base)
    foreGroundImage = cv2.merge([b, g, r, a])

    b, g, r, a = cv2.split(foreGroundImage)
    foreground = cv2.merge((b, g, r))

    alpha = cv2.merge((a, a, a)).astype(float) / 255
    alpha = alpha * factor

    foreground = cv2.multiply(alpha, foreground.astype(float))
    background = cv2.multiply(1 - alpha, background.astype(float))

    outarray = foreground + background

    b, g, r = cv2.split(outarray)
    outarray = cv2.merge([r, g, b])
    outImage = Image.fromarray(np.uint8(outarray))
    return outImage

def process_image(image_path, sticker_image, shape_predictor_path, output_path, magnification):
    """Process an image and save the output with a sticker overlay."""
    ##for 3dmapping uncomment this part
    # try:
    #     initial_image = Image.open(image_path)
    #     zstore = generate_zstore(initial_image)
    #     x, y, w, h, landmarks = get_max_node(image_path)
    #     sticker_with_3d_mapping, y = deformation3d(sticker_image, sticker_image, magnification, zstore, x, y)
    # except IndexError:
        # If IndexError occurs, place the sticker in the middle of the image
    initial_image = Image.open(image_path)
    width, height = initial_image.size
    x = width // 2 - sticker_image.width // 2
    y = height // 2 - sticker_image.height // 2
    sticker_with_3d_mapping = sticker_image

    outImage = make_stick2(backimg=initial_image, sticker=sticker_with_3d_mapping, x=x, y=y, factor=1)
    resized_outImage = outImage.resize((256, 256), Image.Resampling.LANCZOS)
    
    outImage_np = np.array(resized_outImage)
    cv2.imwrite(output_path, cv2.cvtColor(outImage_np, cv2.COLOR_RGB2BGR))

def main():
    # Load the sticker images
    sticker_images = [Image.open(path) for path in STICKER_PATHS]

    # Get all image paths in the input directory
    image_paths = get_all_image_paths(INPUT_DIR)

    # Process each image with each sticker and save the output to the respective directory
    for image_path in image_paths:
        for sticker_image, sticker_name, magnification in zip(sticker_images, STICKER_NAMES, MAGNIFICATION_VALUES):
            output_path = create_output_path(image_path, INPUT_DIR, OUTPUT_DIR, sticker_name)
            process_image(image_path, sticker_image, SHAPE_PREDICTOR_PATH, output_path, magnification=magnification)

if __name__ == "__main__":
    main()
