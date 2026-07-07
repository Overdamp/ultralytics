# 🧠 EX73: Smart Agriculture Grading (Instance Segmentation & Color Analysis)

In smart agriculture, automated grading systems sort produce based on physical characteristics: **size** (volume/diameter) and **ripeness** (coloration). 

Standard bounding boxes are insufficient for grading because they include background pixels and cannot capture irregular shapes. Instead, we use **YOLO Instance Segmentation (YOLO-seg)** to obtain exact pixel-level masks. We then calculate metric sizes from these masks and transform the color space to **HSL (Hue, Saturation, Lightness)** to perform robust ripeness grading.

---

## 1. Principles of Size and Color Analysis

### Precise Size from Binary Masks
A segmentation mask is a binary matrix $M(u,v) \in \{0, 1\}$ representing the object's pixels. We calculate:
*   **Pixel Area ($A_{pixel}$):** The sum of positive mask pixels:
    $$A_{pixel} = \sum_{u} \sum_{v} M(u,v)$$
*   **Metric Size Calibration:** Using a known reference scale (calibration factor $C = \text{mm/pixel}$):
    $$\text{Physical Area (mm}^2) = A_{pixel} \times C^2$$
    $$\text{Physical Diameter (mm)} = \text{Pixel Diameter} \times C$$

### HSL Color Space for Ripeness
RGB values are highly sensitive to lighting changes (shadows, highlights). The **HSL (Hue, Saturation, Lightness)** space decouples color type (chrominance) from brightness (luminance):
*   **Hue ($H \in [0^\circ, 360^\circ]$):** Represents the pure color. For instance:
    *   $60^\circ - 150^\circ$ corresponds to **Green** (unripe).
    *   $15^\circ - 60^\circ$ corresponds to **Yellow/Orange** (ripe).
    *   $0^\circ - 15^\circ$ or $330^\circ - 360^\circ$ corresponds to **Red** (overripe/ripe).
*   **Saturation ($S$):** The intensity/purity of the color.
*   **Lightness ($L$):** The brightness.

By analyzing the Hue histogram of *only* the pixels where $M(u,v) = 1$, we can accurately determine the fruit's ripeness category, independent of illumination changes.

---

## 💻 Python Implementation

The following script simulates receiving a YOLO-seg binary mask and an RGB crop of a mango, calculates its physical size, and determines its ripeness by analyzing the Hue channel.

```python
import numpy as np
import matplotlib.colors as colors

# 1. Generate synthetic fruit data (a yellow-green mango mask & image)
height, width = 200, 200
Y, X = np.ogrid[:height, :width]
center_y, center_x = 100, 100
radius = 60

# Binary mask (circle)
mask = (X - center_x)**2 + (Y - center_y)**2 <= radius**2

# RGB Image: Yellow-green colors inside the mask, black outside
rgb_image = np.zeros((height, width, 3), dtype=np.uint8)
# Ripe yellow-orange is roughly [230, 180, 50], unripe green is [80, 180, 50]
# We'll fill the circle with a gradient representing ripening progress
for y in range(height):
    for x in range(width):
        if mask[y, x]:
            # Gradient: left side is yellow, right side is green
            ratio = x / width
            r = int(230 * (1 - ratio) + 80 * ratio)
            g = int(180)
            b = int(50)
            rgb_image[y, x] = [r, g, b]

# 2. Compute Physical Dimensions
calibration_factor = 0.25  # 1 pixel = 0.25 mm
pixel_area = np.sum(mask)
physical_area_mm2 = pixel_area * (calibration_factor ** 2)

# Diameter calculation (bounding box width from mask)
mask_coords = np.argwhere(mask)
y_min, x_min = mask_coords.min(axis=0)
y_max, x_max = mask_coords.max(axis=0)
pixel_diameter = max(y_max - y_min, x_max - x_min)
physical_diameter_mm = pixel_diameter * calibration_factor

# 3. HSL Color Analysis
# Convert RGB to normalized float [0, 1], then to HSL
rgb_normalized = rgb_image / 255.0
hsv_image = colors.rgb_to_hsv(rgb_normalized)  # In matplotlib/numpy, HSV is commonly used (Hue is equivalent to HSL Hue)

# Extract Hue values for mask pixels only
hue_values = hsv_image[:, :, 0][mask] * 360.0  # Convert scale 0-1 to degrees 0-360

# Classify ripeness based on mean Hue
mean_hue = np.mean(hue_values)

if 60.0 <= mean_hue <= 150.0:
    ripeness_grade = "Unripe (Green)"
elif 20.0 <= mean_hue < 60.0:
    ripeness_grade = "Ripe (Yellow/Orange)"
else:
    ripeness_grade = "Overripe (Red)"

print(f"--- Fruit Grading Report ---")
print(f"Total Pixel Area: {pixel_area} pixels")
print(f"Physical Area: {physical_area_mm2:.2f} mm²")
print(f"Physical Diameter: {physical_diameter_mm:.2f} mm")
print(f"Average Hue Angle: {mean_hue:.1f}°")
print(f"Ripeness Classification: {ripeness_grade}")
```

---

## 💡 Professor Tips

*   **Calibration Markers:** In a real factory conveyor belt system, camera distance is fixed. A camera calibration step using **ArUco markers** or checkerboards is run once to calculate the exact homography and physical distance per pixel.
*   **Why Hue decouples brightness:** Shadows (e.g. from the fruit curvature) make RGB values drop proportionally. However, Hue is calculated as an angle ($H = \arctan(\dots)$ of color differences), meaning scaling R, G, and B by a constant factor (shading) does not change the Hue angle, keeping color detection robust under uneven lighting.

---

*Related Topics:*
*   [[EX72_Autonomous_Driving_BEV]]
*   [[EX74_Thermal_Fire_Detection]]
*   [[YOLO_Learning_Plan]]
*   [[learning_journal]]
