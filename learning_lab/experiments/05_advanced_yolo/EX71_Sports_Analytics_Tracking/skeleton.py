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
        Predicts the next state and update state covariance.
        
        Returns:
            np.ndarray: Predicted 2D coordinates [x, y].
        """
        # TODO: Implement Kalman Filter state and covariance prediction:
        # x_k|k-1 = F * x_k-1|k-1
        # P_k|k-1 = F * P_k-1|k-1 * F^T + Q
        pass
        
    def update(self, z: np.ndarray) -> np.ndarray:
        """
        Updates the state estimate using a new sensor measurement z.
        
        Args:
            z (np.ndarray): Measurement vector [x, y]^T.
            
        Returns:
            np.ndarray: Updated 2D coordinates [x, y].
        """
        # TODO: Implement Kalman Filter measurement update:
        # y = z - H * x_k|k-1       (residual)
        # S = H * P_k|k-1 * H^T + R (residual covariance)
        # K = P_k|k-1 * H^T * S^-1  (Kalman gain)
        # x_k|k = x_k|k-1 + K * y
        # P_k|k = (I - K * H) * P_k|k-1
        pass

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
    # TODO: Perform prediction phase first.
    
    # TODO: Check if a YOLO detection box is available:
    # 1. If available:
    #    a. Extract the center coordinate (x, y) from the bounding box.
    #    b. Update the Kalman Filter using the center coordinate.
    #    c. Return the updated estimated coordinates.
    # 2. If not available (None):
    #    a. Skip the update phase.
    #    b. Return the predicted coordinates.
    pass
