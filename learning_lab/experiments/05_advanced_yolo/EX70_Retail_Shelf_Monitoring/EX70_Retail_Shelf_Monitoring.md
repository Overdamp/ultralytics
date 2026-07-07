# 🧠 EX70: Retail Shelf Monitoring (Few-Shot Product Matching)

In large-scale retail environments, detecting and identifying thousands of constantly changing product stock-keeping units (SKUs) is a major computer vision challenge. Training a standard YOLO model to classify 10,000+ individual products requires massive training datasets and constant retraining whenever a new product is introduced. 

To solve this, we separate the problem into a two-stage pipeline:
1. **Class-Agnostic Detection:** A YOLO model is trained to detect *any* generic product on a shelf, yielding bounding boxes without classifying the specific SKU.
2. **Few-Shot Feature Embedding & Vector Search:** The detected product crops are passed through a feature extractor (e.g., a Siamese or pre-trained ResNet model) to generate low-dimensional dense embeddings. These embeddings are compared against a reference database of target product images using cosine similarity or vector search.

---

## 1. Core Concepts & Mathematical Formulation

### Class-Agnostic YOLO Detection
A class-agnostic YOLO detector predicts bounding boxes ($x, y, w, h$) and a single confidence score indicating the presence of *any* product, rather than calculating individual probability distributions over thousands of classes.

### Feature Extraction via Siamese / ResNet Embeddings
Let a cropped product image be $x$. A deep neural network encoder $f(\cdot)$ maps $x$ to a high-dimensional vector space:

$$\mathbf{z} = f(x) \in \mathbb{R}^d$$

where $d$ is the embedding dimension (typically 128, 256, or 512). The network is trained using loss functions like **Triplet Loss** or **Contrastive Loss** to ensure that images of the same product are close together in the vector space, while images of different products are far apart.

### Cosine Similarity Search
To identify a queried product embedding $\mathbf{z}_q$ against a database of reference embeddings $\mathbf{z}_i$, we compute the **Cosine Similarity**:

$$\text{Similarity}(\mathbf{z}_q, \mathbf{z}_i) = \frac{\mathbf{z}_q \cdot \mathbf{z}_i}{\|\mathbf{z}_q\| \|\mathbf{z}_i\|} = \frac{\sum_{j=1}^d z_{q,j} z_{i,j}}{\sqrt{\sum_{j=1}^d z_{q,j}^2} \sqrt{\sum_{j=1}^d z_{i,j}^2}}$$

The reference SKU with the highest cosine similarity (above a predefined threshold) is chosen as the matched product.

---

## 💻 Python Implementation

The following script simulates the pipeline: it takes mock YOLO product detections, generates feature embeddings, and matches them to a reference gallery database.

```python
import numpy as np

# 1. Simulate embedding database (Gallery) of 3 known SKUs
# Each SKU has a 128-dimensional reference embedding
np.random.seed(42)
embedding_dim = 128
gallery_skus = ["Cola_Classic", "Diet_Cola", "Orange_Soda"]

# Create normalized reference embeddings
gallery_embeddings = {}
for sku in gallery_skus:
    vec = np.random.randn(embedding_dim)
    gallery_embeddings[sku] = vec / np.linalg.norm(vec)

# 2. Simulate detecting a product on a shelf using class-agnostic YOLO
# The detector outputs bounding boxes: [x_min, y_min, x_max, y_max]
detected_boxes = [
    [100, 150, 180, 350],  # Box 1
    [200, 150, 280, 350]   # Box 2
]

# Simulate crop embeddings (adding some noise to the reference embeddings)
query_embeddings = []
# Query 1: A slightly noisy "Cola_Classic"
noise_1 = 0.15 * np.random.randn(embedding_dim)
q_vec_1 = gallery_embeddings["Cola_Classic"] + noise_1
query_embeddings.append(q_vec_1 / np.linalg.norm(q_vec_1))

# Query 2: A slightly noisy "Orange_Soda"
noise_2 = 0.20 * np.random.randn(embedding_dim)
q_vec_2 = gallery_embeddings["Orange_Soda"] + noise_2
query_embeddings.append(q_vec_2 / np.linalg.norm(q_vec_2))

# 3. Perform Cosine Similarity Search
similarity_threshold = 0.70

for idx, q_emb in enumerate(query_embeddings):
    print(f"\nAnalyzing detected product crop {idx + 1} at bounding box {detected_boxes[idx]}:")
    
    best_match = None
    best_score = -1.0
    
    # Compare with each SKU in the gallery
    for sku, ref_emb in gallery_embeddings.items():
        cosine_sim = np.dot(q_emb, ref_emb)
        print(f" - Similarity with {sku}: {cosine_sim:.4f}")
        
        if cosine_sim > best_score:
            best_score = cosine_sim
            best_match = sku
            
    # Decision logic
    if best_score >= similarity_threshold:
        print(f"🎯 MATCHED: SKU classified as '{best_match}' (Confidence: {best_score:.4f})")
    else:
        print(f"⚠️ UNKNOWN: No gallery SKU exceeded the threshold (Best: {best_match} at {best_score:.4f})")
```

---

## 💡 Professor Tips

*   **Why Hashing & Vector DBs?** In a real supermarket with 50,000 SKUs, performing a linear scan (brute-force search) across all reference images for every shelf crop is too slow. Engineers use vector indexing databases such as **FAISS**, **Milvus**, or **Qdrant**, which use Approximate Nearest Neighbor (ANN) search algorithms (like HNSW) to reduce lookup times to sub-milliseconds.
*   **Agnostic Detector Fine-tuning:** To train a class-agnostic YOLO detector, collapse all dataset labels (e.g., different product brands) into a single generic category named `product`. This simplifies object detection optimization and lets the model focus purely on bounding box localization.

---

*Related Topics:*
*   [[EX71_Sports_Analytics_Tracking]]
*   [[YOLO_Learning_Plan]]
*   [[learning_journal]]
