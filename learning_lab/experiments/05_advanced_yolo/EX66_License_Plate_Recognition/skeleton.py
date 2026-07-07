import cv2
import numpy as np
from typing import Any

def preprocess_plate_crop(crop_image: np.ndarray) -> np.ndarray:
    """
    Applies image preprocessing to optimize character readability for OCR engines.
    
    The steps include:
    1. Grayscaling: Converting the BGR crop to single-channel gray.
    2. Resizing: Upscaling the crop (typically by 2x) to provide higher character resolution.
    3. Adaptive Thresholding: Converting the gray image into a binary image using Gaussian adaptive thresholding
       to separate characters from the background, even under uneven lighting.
       
    Args:
        crop_image (np.ndarray): BGR image crop containing the license plate.
        
    Returns:
        np.ndarray: Binary preprocessed image.
    """
    # TODO: Convert the crop to grayscale
    
    # TODO: Resize (upscale) the grayscale image by 2x using cubic interpolation
    
    # TODO: Apply adaptive thresholding (cv2.adaptiveThreshold)
    # Hint: Use ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, block size of 11, and C of 2
    
    # TODO: Return the preprocessed image
    pass

def read_plate_text(ocr_model: Any, processed_crop: np.ndarray) -> str:
    """
    Performs OCR on a preprocessed license plate crop using the provided OCR model.
    
    This function detects and formats alphanumeric text from the preprocessed crop.
    It supports multiple OCR engines (such as EasyOCR, PyTesseract, or custom models)
    by inspect-routing the model calls.
    
    Args:
        ocr_model (Any): The OCR engine instance.
        processed_crop (np.ndarray): Binary preprocessed plate image.
        
    Returns:
        str: Cleaned alphanumeric plate string.
    """
    # TODO: Detect the OCR engine type or API
    # 1. EasyOCR (has readtext method): call readtext and join the words
    # 2. PyTesseract (has image_to_string method): call image_to_string with config for single line text (--psm 7)
    # 3. Custom Callable: call directly with the crop
    
    # TODO: Clean the text (strip whitespaces) and return it
    pass
