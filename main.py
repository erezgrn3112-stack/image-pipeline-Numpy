import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def run_pipeline(image_path):
    # 1. Load Image & Convert to Array
    try:
        raw_image = Image.open(image_path)
        image = np.array(raw_image)
        print(f"✅ Image loaded successfully. Shape: {image.shape}")
    except FileNotFoundError:
        print("❌ Error: Image file not found. Please check the filename.")
        return

    # --- NumPy Data Augmentation Pipeline ---

    # A. Center Cropping (Dynamic)
    # Instead of hardcoding pixels, we calculate the center 50% of the image
    h, w = image.shape[:2]
    c_h, c_w = h // 2, w // 2  # Center coordinates
    offset_h, offset_w = h // 4, w // 4  # Half of the target window

    # Slicing the array around the center
    cropped_img = image[c_h - offset_h: c_h + offset_h, c_w - offset_w: c_w + offset_w]

    # B. Brightness Adjustment (Broadcasting)
    # Using NumPy broadcasting to add scalar value to the entire matrix instantly
    brightness_factor = 40
    # np.clip ensures pixel values stay valid (0-255) to prevent overflow artifacts
    bright_img = np.clip(image.astype(np.int16) + brightness_factor, 0, 255).astype(np.uint8)

    # C. Horizontal Flip
    # Manipulation of the column index (axis 1)
    flipped_img = np.fliplr(image)

    # D. Grayscale Conversion (Channel Aggregation)
    # Collapsing the 3rd dimension (RGB channels) by calculating the mean
    gray_img = np.mean(image, axis=2).astype(np.uint8)

    # --- Visualization ---
    plt.figure(figsize=(12, 6))
    plt.suptitle("NumPy Data Augmentation Engine", fontsize=16)

    tasks = [
        (image, 'Original\n(Raw Matrix)'),
        (cropped_img, 'Center Crop\n(Slicing)'),
        (bright_img, f'Brightness +{brightness_factor}\n(Broadcasting)'),
        (flipped_img, 'Horizontal Flip\n(Axis Manipulation)'),
        (gray_img, 'Grayscale\n(Channel Mean)')
    ]

    for i, (img, title) in enumerate(tasks):
        plt.subplot(1, 5, i + 1)
        cmap = 'gray' if img.ndim == 2 else None
        plt.imshow(img, cmap=cmap)
        plt.title(title, fontsize=10)
        plt.axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Ensure you have a file named 'test_image.jpg' in the same folder
    run_pipeline('test_image.jpg')