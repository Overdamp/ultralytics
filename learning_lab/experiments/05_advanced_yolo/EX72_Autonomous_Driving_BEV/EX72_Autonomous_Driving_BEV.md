# 🧠 EX72: Autonomous Driving BEV (Bird's Eye View Projection)

In autonomous driving, cameras capture the road in a 3D perspective projection. However, path planning, obstacle avoidance, and control algorithms operate in a 2D ground plane representation known as the **Bird's Eye View (BEV)**. 

To bridge this gap, we detect vehicles and obstacles using 2D object detection (YOLO), locate the contact point where the object touches the ground (bottom-center of the bounding box), and apply **Inverse Perspective Mapping (IPM)** to project these pixel coordinates onto a 2D metric ground grid.

---

## 1. Mathematical Formulation & Coordinate Systems

### Perspective Projection Model
The relationship between a 3D world coordinate point $\mathbf{X}_w = [X_w, Y_w, Z_w]^T$ and its 2D pixel projection $\mathbf{x}_p = [u, v]^T$ is governed by the camera intrinsic matrix $\mathbf{K}$ and the extrinsic matrix $[\mathbf{R} | \mathbf{t}]$:

$$s \begin{bmatrix} u \\ v \\ 1 \end{bmatrix} = \mathbf{K} \begin{bmatrix} \mathbf{R} & \mathbf{t} \end{bmatrix} \begin{bmatrix} X_w \\ Y_w \\ Z_w \\ 1 \end{bmatrix}$$

Where:
*   $\mathbf{K} = \begin{bmatrix} f_x & 0 & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1 \end{bmatrix}$ is the camera intrinsic matrix (focal lengths and optical center).
*   $\mathbf{R}$ (Rotation matrix) and $\mathbf{t}$ (Translation vector) represent the extrinsic matrix, defining the camera's position and orientation relative to the road/vehicle coordinate system.
*   $s$ is an arbitrary scaling factor representing depth.

### Inverse Perspective Mapping (IPM)
Because projecting a 2D pixel back to 3D is mathematically underdetermined (we lose depth along the ray of projection), we must introduce a geometric constraint. In road scenarios, we assume the ground is flat:

$$Z_w = 0$$

Let $\mathbf{M} = \mathbf{K} \begin{bmatrix} \mathbf{r}_1 & \mathbf{r}_2 & \mathbf{t} \end{bmatrix}$ be a $3\times3$ homography matrix mapping the ground plane to the image, where $\mathbf{r}_1$ and $\mathbf{r}_2$ are the first two columns of $\mathbf{R}$. We can write:

$$\begin{bmatrix} u \\ v \\ 1 \end{bmatrix} \sim \mathbf{H}_{ground} \begin{bmatrix} X_w \\ Y_w \\ 1 \end{bmatrix} \quad \text{where} \quad \mathbf{H}_{ground} = \mathbf{K} \begin{bmatrix} \mathbf{r}_1 & \mathbf{r}_2 & \mathbf{t} \end{bmatrix}$$

By computing the inverse homography matrix $\mathbf{H}_{ground}^{-1}$, we project any pixel coordinate back to the 2D road plane:

$$\begin{bmatrix} X_w \\ Y_w \\ 1 \end{bmatrix} \sim \mathbf{H}_{ground}^{-1} \begin{bmatrix} u \\ v \\ 1 \end{bmatrix}$$

---

## 💻 Python Implementation

The following script sets up simulated camera matrices, takes a YOLO bounding box prediction of a vehicle, extracts its bottom-center pixel, and projects it to a BEV ground plane coordinate in meters.

```python
import numpy as np

# 1. Camera Intrinsics (K)
focal_length_px = 800  # fx, fy
img_w, img_h = 1920, 1080
cx, cy = img_w / 2, img_h / 2
K = np.array([
    [focal_length_px, 0, cx],
    [0, focal_length_px, cy],
    [0, 0, 1]
])

# 2. Camera Extrinsics (R, t) - Camera mounted 1.5 meters high, tilted down by 15 degrees
pitch_deg = 15.0
pitch_rad = np.radians(pitch_deg)
c, s = np.cos(pitch_rad), np.sin(pitch_rad)

# Rotation matrix around X-axis (tilting down)
R = np.array([
    [1,  0, 0],
    [0,  c, s],
    [0, -s, c]
])
# Translation vector (1.5 meters above ground)
t = np.array([[0.0], [1.5], [0.0]])

# 3. Form Ground Homography Matrix H_ground (for Z_w = 0)
r1 = R[:, 0:1]
r2 = R[:, 1:2]
H_ground = K.dot(np.hstack((r1, r2, t)))
H_inv = np.linalg.inv(H_ground)

# 4. Simulate YOLO Vehicle Bounding Box Output: [x_min, y_min, x_max, y_max]
# Let's say a vehicle is detected directly in front of the camera
yolo_bbox = [860, 600, 1060, 800]

# Bottom-center of the bounding box represents the contact point on the road
u_bottom = (yolo_bbox[0] + yolo_bbox[2]) / 2.0  # (860+1060)/2 = 960
v_bottom = yolo_bbox[3]                          # 800 (bottom of the box)
pixel_coord = np.array([[u_bottom], [v_bottom], [1.0]])

# 5. Project back to 2D Ground Plane (IPM)
ground_homogeneous = H_inv.dot(pixel_coord)
# Normalize to remove scaling factor s
ground_meters = ground_homogeneous / ground_homogeneous[2, 0]

X_w = ground_meters[0, 0]
Y_w = ground_meters[1, 0]

print(f"YOLO Bounding Box: {yolo_bbox}")
print(f"Road Contact Pixel (u, v): ({u_bottom:.1f}, {v_bottom:.1f})")
print("-" * 50)
print(f"Projected 2D Ground Coordinates (BEV relative to camera base):")
print(f"Lateral Distance (X_w): {X_w:.3f} meters (positive is right, negative is left)")
print(f"Longitudinal Distance (Y_w): {Y_w:.3f} meters (distance ahead on the road)")
```

---

## 💡 Professor Tips

*   **Road slope and Pitch issues:** IPM is highly sensitive to the flat-ground assumption. If the road slopes up or down, or if the ego-vehicle accelerates or brakes (causing pitch rotation), the projected distances will suffer from massive errors (e.g. projecting a car at 30 meters as if it were at 50 meters). Autonomous vehicle stacks integrate **Inertial Measurement Units (IMUs)** and suspension sensors to update the extrinsic rotation matrix $\mathbf{R}$ in real-time.
*   **Bounding Box Anchor Selection:** When projecting vehicles, always project the *bottom-center* of the bounding box. Avoid projecting the center of the bounding box, as the center floats in 3D space above the ground, violating the $Z_w = 0$ constraint and causing the object to be projected too far away.

---

*Related Topics:*
*   [[EX71_Sports_Analytics_Tracking]]
*   [[EX73_Smart_Agriculture_Grading]]
*   [[YOLO_Learning_Plan]]
*   [[learning_journal]]
