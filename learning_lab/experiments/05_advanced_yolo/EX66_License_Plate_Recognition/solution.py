import cv2
import numpy as np
from typing import Any

def preprocess_plate_crop(crop_image: np.ndarray) -> np.ndarray:
    """
    Applies image preprocessing to optimize character readability for OCR engines.
    
    The steps include:
    1. Grayscaling: Converting the BGR crop to single-channel gray.
    2. Resizing: Upscaling the crop by 2x to provide higher character resolution.
    3. Adaptive Thresholding: Converting the gray image into a binary image using Gaussian adaptive thresholding.
       
    Args:
        crop_image (np.ndarray): BGR image crop containing the license plate.
        
    Returns:
        np.ndarray: Binary preprocessed image.
    """
    if crop_image is None or crop_image.size == 0:
        return np.array([], dtype=np.uint8)
        
    # Convert to grayscale
    gray = cv2.cvtColor(crop_image, cv2.COLOR_BGR2GRAY)
    
    # Resize (upscale by 2x) to increase character pixels
    resized = cv2.resize(gray, (0, 0), fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    
    # Apply Gaussian Adaptive Thresholding
    processed = cv2.adaptiveThreshold(
        resized,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        2
    )
    
    return processed

def read_plate_text(ocr_model: Any, processed_crop: np.ndarray) -> str:
    """
    Performs OCR on a preprocessed license plate crop using the provided OCR model.
    
    Args:
        ocr_model (Any): The OCR engine instance.
        processed_crop (np.ndarray): Binary preprocessed plate image.
        
    Returns:
        str: Cleaned alphanumeric plate string.
    """
    if processed_crop is None or processed_crop.size == 0:
        return ""
        
    text = ""
    
    # Check if ocr_model is EasyOCR (has readtext method)
    if hasattr(ocr_model, 'readtext'):
        results = ocr_model.readtext(processed_crop)
        # Results format: List of Tuple[bbox, text, confidence]
        text = " ".join([res[1] for res in results])
    # Check if ocr_model is a pytesseract-like module/object
    elif hasattr(ocr_model, 'image_to_string'):
        custom_config = r'--oem 3 --psm 7'
        text = ocr_model.image_to_string(processed_crop, config=custom_config)
    # Check if ocr_model is callable (e.g. wrapper function)
    elif callable(ocr_model):
        text = ocr_model(processed_crop)
    # Check if we can fallback to standard pytesseract module
    else:
        try:
            import pytesseract
            custom_config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-'
            text = pytesseract.image_to_string(processed_crop, config=custom_config)
        except ImportError:
            raise ValueError(
                "Provided ocr_model is not recognized and pytesseract is not installed."
            )
            
    return text.strip()
