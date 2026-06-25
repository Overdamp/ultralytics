# 🧠 EX40: Feature Embeddings in Computer Vision

An embedding is a low-dimensional vector representation of high-dimensional data (like images or bounding boxes). Features that are semantically similar (e.g., two different types of valves) are mapped to points that are close to each other in this continuous vector space.

---

## 1. Dimensionality Reduction in CNNs
Raw image patches contain massive numbers of pixels. For instance, a small $128 \times 128$ RGB crop of a valve contains:
$$128 \times 128 \times 3 = 49,152 \text{ values}$$

Passing this crop through a CNN backbone (like the CSPDarknet inside YOLO) downsamples the image spatially while extracting semantic features, outputting a compact 1D vector (e.g., $128$, $256$, or $512$ dimensions). This dense vector is the **feature embedding**.

---

## 2. Mathematical Similarity Metrics

To compare two embeddings $\mathbf{u}$ and $\mathbf{v}$, we use distance or similarity metrics:

### A. Cosine Similarity
Cosine similarity measures the cosine of the angle between two vectors. It ranges from -1 (opposite) to 1 (identical direction), focusing on vector orientation rather than magnitude:

$$\text{Similarity}(\mathbf{u}, \mathbf{v}) = \cos(\theta) = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\| \|\mathbf{v}\|} = \frac{\sum_{i=1}^{d} u_i v_i}{\sqrt{\sum_{i=1}^{d} u_i^2} \sqrt{\sum_{i=1}^{d} v_i^2}}$$

### B. Euclidean Distance ($L2$ Norm)
Euclidean distance measures the straight-line distance between two points in $d$-dimensional space:

$$d(\mathbf{u}, \mathbf{v}) = \|\mathbf{u} - \mathbf{v}\|_2 = \sqrt{\sum_{i=1}^{d} (u_i - v_i)^2}$$

---

## 🔬 Computer Vision Case Study: Multi-Object Tracking (DeepSORT)
In YOLO-based tracking pipelines (like DeepSORT or BoT-SORT used in YOLO tracking modes), embeddings are crucial for **Re-Identification (Re-ID)**:

1.  **Frame 1:** YOLO detects a `control-valve` and extracts its 128-dimensional embedding vector $\mathbf{u}_1$.
2.  **Frame 2:** The valve is temporarily occluded behind a pipe.
3.  **Frame 3:** YOLO detects the valve again and extracts embedding $\mathbf{u}_2$.
4.  **Association:** The tracker computes the Cosine Similarity between $\mathbf{u}_2$ and all active tracks. Since $\text{Similarity}(\mathbf{u}_1, \mathbf{u}_2) = 0.94$, the system confirms it is the same valve and preserves its Track ID, even though the coordinates shifted.

---

## 💻 Python Similarity Calculation (NumPy)

```python
import numpy as np

def cosine_similarity(u, v):
    dot_product = np.dot(u, v)
    norm_u = np.linalg.norm(u)
    norm_v = np.linalg.norm(v)
    if norm_u == 0 or norm_v == 0:
        return 0.0
    return dot_product / (norm_u * norm_v)

# Example 128-d embedding vectors
emb_valve1 = np.random.rand(128)
emb_valve2 = emb_valve1 + np.random.normal(0, 0.05, 128)  # Similar vector
emb_flange = np.random.rand(128)  # Unrelated vector

sim_valve = cosine_similarity(emb_valve1, emb_valve2)
sim_flange = cosine_similarity(emb_valve1, emb_flange)

print(f"Similarity (Valve 1 vs Valve 2): {sim_valve:.4f}")  # Should be close to 1.0
print(f"Similarity (Valve 1 vs Flange): {sim_flange:.4f}")
```

---
*Related Topics:*
*   [[EX39_Fine_Tuning\|EX39: Fine-Tuning]]
*   [[EX41_Bounding_Box\|EX41: Bounding Box Coordinates]]
*   Return to main plan: [[YOLO_Learning_Plan]]
*   Progress log: [[learning_journal]]
