import numpy as np

def cosine_similarity(u, v):
    """
    Calculate the cosine similarity between two 1D vectors u and v.
    """
    dot_product = np.dot(u, v)
    norm_u = np.linalg.norm(u)
    norm_v = np.linalg.norm(v)
    
    if norm_u == 0.0 or norm_v == 0.0:
        return 0.0
        
    return float(dot_product / (norm_u * norm_v))

def euclidean_distance(u, v):
    """
    Calculate the L2 Euclidean distance between two 1D vectors u and v.
    """
    return float(np.linalg.norm(u - v))

if __name__ == "__main__":
    u = np.array([1.0, 2.0, 3.0])
    v = np.array([1.1, 1.9, 3.1])
    
    sim = cosine_similarity(u, v)
    dist = euclidean_distance(u, v)
    
    print("--- Training Results ---")
    print(f"Cosine Similarity : {sim:.6f}")
    print(f"Euclidean Distance: {dist:.6f}")
