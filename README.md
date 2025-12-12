# NumPy Data Augmentation Pipeline

An efficient, vectorized image processing pipeline built purely with **NumPy**.
This project demonstrates how to manipulate image data (tensors) mathematically without using slow Python loops.

## Key Concepts Implemented
* **Vectorization & Broadcasting:** Adjusting pixel brightness across the entire image matrix instantly.
* **Dynamic Slicing:** Algorithmically calculating crop coordinates based on image dimensions (not hardcoded).
* **Dimensionality Reduction:** Converting RGB tensors to Grayscale via channel averaging (Axis aggregation).
* **Type Safety:** Managing `uint8` vs `int16` data types to prevent pixel overflow artifacts.

## Tech Stack
* **Python 3.x**
* **NumPy** (Core matrix operations)
* **Matplotlib** (Visualization)
* **Pillow** (I/O only)

## How to Run
1. Clone the repo.
2. Place an image named `test_image.jpg` in the root folder.
3. Run the script:
   ```bash
   python main.py