# 🧠 EX74: Thermal Fire Detection (Dual-Camera Sensor Fusion)

Using RGB cameras for fire and smoke detection is highly prone to false positives. Visual signatures like orange safety vests, red emergency lights, sun reflections, or yellow billboards can easily trick a YOLO model into triggering false alarms. 

To create a robust industrial warning system, we implement **co-registered dual-camera sensor fusion**. We combine high-resolution RGB video with a Long-Wave Infrared (LWIR) thermal camera. The pipeline runs YOLO on the RGB stream, projects the detected bounding boxes onto the co-aligned thermal grid using a **Homography Matrix**, and analyzes the thermal intensity values to verify if an actual high-heat signature is present.

---

## 1. Principles of Co-Registration & Homography Fusion

### Camera Offset (Parallax) and Homography
Because the RGB and thermal lenses are located at slightly different physical coordinates, their viewpoints do not align perfectly. This spatial offset is called **parallax**. 

If the scene is relatively flat or far away from the cameras, we can model the coordinate transform between the two image planes using a $3\times3$ **Homography Matrix** $\mathbf{H}$:

$$\mathbf{x}_{thermal} \sim \mathbf{H} \mathbf{x}_{rgb}$$

Written out in homogeneous coordinates:

$$\begin{bmatrix} x_t \\ y_t \\ 1 \end{bmatrix} \sim \begin{bmatrix} h_{11} & h_{12} & h_{13} \\ h_{21} & h_{22} & h_{23} \\ h_{31} & h_{32} & h_{33} \end{bmatrix} \begin{bmatrix} x_r \\ y_r \\ 1 \end{bmatrix}$$

Applying the projection:

$$x_t = \frac{h_{11}x_r + h_{12}y_r + h_{13}}{h_{31}x_r + h_{32}y_r + h_{33}}, \quad y_t = \frac{h_{21}x_r + h_{22}y_r + h_{23}}{h_{31}x_r + h_{32}y_r + h_{33}}$$

### Alarm Verification Pipeline
1. **Detection:** Run YOLO on the RGB frame to detect candidates labeled `fire` or `smoke`, outputting bounding box coordinates $[x_1, y_1, x_2, y_2]$.
2. **Coordinate Projection:** Project the corner points of the RGB bounding box into the thermal frame coordinates using $\mathbf{H}$.
3. **Thermal Query:** Extract thermal pixel values from the corresponding projected area in the LWIR frame.
4. **Decision Rules:** Verify if the maximum temperature within the region exceeds a safety threshold ($T_{threshold} \ge 80^\circ\text{C}$). If not, dismiss the detection as a false positive.

---

## 💻 Python Implementation

The following script simulates the fusion pipeline: projecting a detected RGB bounding box onto a co-aligned thermal image grid and evaluating the temperature statistics.

```python
import numpy as np

# 1. Define the RGB-to-Thermal Homography Matrix (H)
# Calculated during camera calibration
H = np.array([
    [0.85, 0.02, 10.0],
    [-0.01, 0.84, 15.0],
    [0.0001, -0.0002, 1.0]
])

def project_point(x, y, homography):
    """Projects a single 2D point from RGB to Thermal coordinates."""
    point = np.array([[x], [y], [1.0]])
    projected = homography.dot(point)
    # Convert from homogeneous coordinates
    x_proj = projected[0, 0] / projected[2, 0]
    y_proj = projected[1, 0] / projected[2, 0]
    return int(round(x_proj)), int(round(y_proj))

# 2. Simulate YOLO RGB Detection of a "fire" candidate
# Bounding box coordinates: [x_min, y_min, x_max, y_max]
rgb_bbox = [200, 150, 300, 250]

# Project the top-left and bottom-right corners to the thermal frame
tx_min, ty_min = project_point(rgb_bbox[0], rgb_bbox[1], H)
tx_max, ty_max = project_point(rgb_bbox[2], rgb_bbox[3], H)

print(f"YOLO RGB Detection Bounding Box: {rgb_bbox}")
print(f"Projected Thermal Bounding Box : [{tx_min}, {ty_min}, {tx_max}, {ty_max}]")
print("-" * 65)

# 3. Simulate two scenarios for the corresponding thermal image crop (e.g. 400x400 grid)
# Scenario A: False alarm (a hot lamp or red balloon - visual matches, but temperature is 35°C)
thermal_grid_A = np.random.uniform(20.0, 35.0, size=(400, 400)) 

# Scenario B: True fire (temperature in the center of the crop reaches 250°C)
thermal_grid_B = np.random.uniform(20.0, 35.0, size=(400, 400))
# Inject heat values in the projected area
thermal_grid_B[ty_min:ty_max, tx_min:tx_max] = np.random.uniform(150.0, 300.0, size=(ty_max-ty_min, tx_max-tx_min))

# 4. Verification Logic
temp_threshold_celsius = 80.0

for name, grid in [("Scenario A (Red Balloon)", thermal_grid_A), ("Scenario B (Actual Fire)", thermal_grid_B)]:
    # Crop the projected region from the thermal grid
    # Ensure crop coordinates do not exceed grid boundaries
    h_max, w_max = grid.shape
    crop = grid[max(0, ty_min):min(h_max, ty_max), max(0, tx_min):min(w_max, tx_max)]
    
    max_temp = np.max(crop)
    mean_temp = np.mean(crop)
    
    print(f"\nEvaluating {name}:")
    print(f" - Max Crop Temp: {max_temp:.1f}°C")
    print(f" - Mean Crop Temp: {mean_temp:.1f}°C")
    
    if max_temp >= temp_threshold_celsius:
        print(f"🔥 ALARM CONFIRMED: High temperature signature verified.")
    else:
        print(f"🟢 FALSE POSITIVE DISMISSED: Temperature within normal limits.")
```

---

## 💡 Professor Tips

*   **Parallax Limits:** Homography registration works perfectly only when all targets are on a single plane (e.g., ground plane) or far away (where camera baseline distance is negligible). For close-range targets with depth variation, homography mapping breaks down. In those cases, engineers use **Stereo Matching** or **Active Depth Sensors** to align RGB and LWIR channels pixel-by-pixel.
*   **Thermal Raw Data Calibration:** Most industrial thermal cameras output raw 14-bit digital numbers (Radiometric data) instead of Celsius. You must convert these raw digital numbers to Kelvin or Celsius using the camera's calibration parameters (emissivity, atmospheric temperature, and distance) before running the threshold logic.

---

*Related Topics:*
*   [[EX73_Smart_Agriculture_Grading]]
*   [[YOLO_Learning_Plan]]
*   [[learning_journal]]
