import numpy as np
import cv2
import torch
import torch.nn as nn
from typing import List, Dict, Tuple, Optional

def match_products_to_database(
    detected_crops: List[np.ndarray],
    embedding_model: nn.Module,
    vector_database: Dict[str, np.ndarray],
    threshold: float = 0.70
) -> List[Tuple[Optional[str], float]]:
    """
    Extracts feature embeddings for detected product crops and queries a vector 
    database using cosine similarity to identify product SKUs.

    This function simulates a few-shot product matching pipeline commonly used in 
    retail shelf monitoring. Since training a YOLO classifier for thousands of constantly 
    changing SKUs is impractical, this two-stage approach uses a class-agnostic YOLO 
    detector to find product crops, and an embedding model combined with a vector 
    database to identify them.

    Args:
        detected_crops (List[np.ndarray]): A list of product image crops (each HxWxC, RGB format).
        embedding_model (nn.Module): A pre-trained feature extractor (e.g., ResNet or Siamese network)
                                     that outputs a 1D feature tensor for a batch of images.
        vector_database (Dict[str, np.ndarray]): A dictionary mapping SKU names (keys) to their 
                                                 normalized 1D reference embeddings (values).
        threshold (float): The minimum cosine similarity score required to confirm a match.
                           Default is 0.70.

    Returns:
        List[Tuple[Optional[str], float]]: A list of tuples, one for each crop, containing:
                                           - The matched SKU name (str) or None if no SKU 
                                             exceeded the similarity threshold.
                                           - The highest cosine similarity score (float) achieved.
    """
    # Put the embedding model in evaluation mode
    embedding_model.eval()
    
    matches = []
    
    with torch.no_grad():
        for crop in detected_crops:
            # 1. Resize to the standard input dimensions expected by the embedding model (e.g., 224x224)
            # Ensure crop is a valid image with H, W > 0
            if crop.size == 0 or crop.shape[0] == 0 or crop.shape[1] == 0:
                matches.append((None, -1.0))
                continue
                
            crop_resized = cv2.resize(crop, (224, 224))
            
            # 2. Normalize pixel values to [0, 1]
            if crop_resized.dtype == np.uint8:
                crop_resized = crop_resized.astype(np.float32) / 255.0
            
            # Standard ImageNet normalization: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            crop_normalized = (crop_resized - mean) / std
            
            # 3. Convert the crop to a PyTorch tensor, rearrange channels to CHW, and add a batch dimension
            crop_tensor = torch.from_numpy(crop_normalized).permute(2, 0, 1).unsqueeze(0)
            
            # Use the same device as the model parameters if possible
            device = next(embedding_model.parameters()).device if list(embedding_model.parameters()) else torch.device("cpu")
            crop_tensor = crop_tensor.to(device)
            
            # Pass the tensor through the embedding model to extract the feature representation
            embedding_tensor = embedding_model(crop_tensor)
            
            # Convert the extracted embedding to a 1D NumPy array
            embedding_np = embedding_tensor.squeeze().cpu().numpy()
            
            # Normalize it to unit length (L2 norm)
            norm = np.linalg.norm(embedding_np)
            if norm > 0:
                query_vector = embedding_np / norm
            else:
                query_vector = embedding_np
                
            # Query the vector database using cosine similarity
            best_sku = None
            best_score = -1.0
            
            for sku, ref_embedding in vector_database.items():
                # Ensure the reference embedding is normalized
                ref_norm = np.linalg.norm(ref_embedding)
                if ref_norm > 0:
                    ref_vector = ref_embedding / ref_norm
                else:
                    ref_vector = ref_embedding
                
                # Cosine similarity between normalized vectors is their dot product
                similarity = float(np.dot(query_vector, ref_vector))
                
                if similarity > best_score:
                    best_score = similarity
                    best_sku = sku
            
            # Apply threshold logic
            if best_score >= threshold:
                matches.append((best_sku, best_score))
            else:
                matches.append((None, best_score))
                
    return matches
