import numpy as np
from typing import List, Tuple, Optional

class KalmanFilter2D:
    """
    A simple 2D Kalman Filter for tracking position (x, y) and velocity (vx, vy)
    of an object in a 2D plane under a constant velocity model.
    """
    def __init__(self, dt: float = 1.0, process_noise: float = 0.1, measurement_noise: float = 2.0):
        # State vector [x, y, vx, vy]^T
        self.x = np.zeros((4, 1), dtype=np.float32)
        
        # State transition matrix F
        self.F = np.array([
            [1.0, 0.0,  dt, 0.0],
            [0.0, 1.0, 0.0,  dt],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ], dtype=np.float32)
        
        # Measurement matrix H
        self.H = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0]
        ], dtype=np.float32)
        
        # Covariance matrices
        self.P = np.eye(4, dtype=np.float32) * 10.0
        self.Q = np.eye(4, dtype=np.float32) * process_noise
        self.R = np.eye(2, dtype=np.float32) * measurement_noise
        
    def predict(self) -> np.ndarray:
        """
        Predicts the next state and updates state covariance.
        
        Returns:
            np.ndarray: Predicted 2D coordinates [x, y].
        """
        self.x = np.dot(self.F, self.x)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        return self.x[:2].flatten()
        
    def update(self, z: np.ndarray) -> np.ndarray:
        """
        Updates the state estimate using a new sensor measurement z.
        
        Args:
            z (np.ndarray): Measurement vector [x, y]^T.
            
        Returns:
            np.ndarray: Updated 2D coordinates [x, y].
        """
        z = np.array(z, dtype=np.float32).reshape(2, 1)
        y = z - np.dot(self.H, self.x)  # Innovation/Residual
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))  # Kalman Gain
        
        # Update State and Covariance
        self.x = self.x + np.dot(K, y)
        self.P = np.dot(np.eye(4) - np.dot(K, self.H), self.P)
        return self.x[:2].flatten()

def update_ball_tracker(
    kalman_filter: KalmanFilter2D,
    detection_box: Optional[List[float]]
) -> Tuple[float, float]:
    """
    Updates the Kalman Filter tracker with the current frame's YOLO detection box.
    
    If a detection box is available, the filter predicts and then updates its state 
    using the center of the box. If the box is missing (due to occlusion or blur), 
    the filter relies solely on prediction to estimate the position.

    Args:
        kalman_filter (KalmanFilter2D): The Kalman Filter instance tracking the object.
        detection_box (Optional[List[float]]): Bounding box coordinates from YOLO 
                                               in format [x_min, y_min, x_max, y_max], 
                                               or None if the object is not detected.

    Returns:
        Tuple[float, float]: Estimated (x, y) coordinates of the ball.
    """
    # 1. Perform prediction phase first
    predicted_pos = kalman_filter.predict()
    
    # 2. Check if a YOLO detection box is available
    if detection_box is not None:
        # Extract the center coordinate (x, y) from the bounding box
        x_min, y_min, x_max, y_max = detection_box
        x_center = (x_min + x_max) / 2.0
        y_center = (y_min + y_max) / 2.0
        z = np.array([x_center, y_center], dtype=np.float32)
        
        # Update the Kalman Filter using the center coordinate
        final_estimate = kalman_filter.update(z)
        return float(final_estimate[0]), float(final_estimate[1])
    else:
        # If not available, rely purely on the predicted coordinates
        return float(predicted_pos[0]), float(predicted_pos[1])
