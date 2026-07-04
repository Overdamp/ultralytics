import numpy as np

def cosine_similarity(u, v):
    """
    Calculate the cosine similarity between two 1D vectors u and v.
    """
    # TODO: Implement Cosine Similarity: dot(u, v) / (norm(u) * norm(v))
    # Hint: Use np.dot and np.linalg.norm. Handle division by zero.
    return 0.0

def euclidean_distance(u, v):
    """
    Calculate the L2 Euclidean distance between two 1D vectors u and v.
    """
    # TODO: Implement Euclidean Distance: norm(u - v)
    # Hint: Use np.linalg.norm
    return 0.0

if __name__ == "__main__":
    u = np.array([1.0, 2.0, 3.0])
    v = np.array([1.1, 1.9, 3.1])
    
    sim = cosine_similarity(u, v)
    dist = euclidean_distance(u, v)
    
    print("--- Training Results ---")
    if sim != 0.0 or dist != 0.0:
        print(f"Cosine Similarity : {sim:.6f}")
        print(f"Euclidean Distance: {dist:.6f}")
    else:
        print("Embedding metrics logic not implemented yet.")
