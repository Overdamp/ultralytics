"""
🧠 EX50: Hyperparameter Tuning for YOLO (Solution)

This script provides complete implementations for:
1. Cosine Annealing Learning Rate Scheduler.
2. Genetic Algorithm (GA) Mutation function.

It also runs automated print checks to verify correctness.

Author: Computer Vision Professor (Educator Persona)
"""

import math
import random
from typing import Dict, Tuple

def get_cosine_lr(epoch: int, total_epochs: int, lr0: float, lrf: float) -> float:
    """
    Calculates the learning rate for a given epoch using Cosine Annealing.
    
    Formula:
        eta_t = eta_min + 0.5 * (eta_max - eta_min) * (1 + cos(T_cur / T_max * pi))
        
    Where:
        eta_max = lr0
        eta_min = lr0 * lrf
        T_cur = current epoch (0-indexed)
        T_max = total training epochs
        
    Args:
        epoch (int): The current training epoch (0-indexed).
        total_epochs (int): The total number of epochs in training.
        lr0 (float): The initial learning rate (maximum).
        lrf (float): The final learning rate fraction (fraction of lr0).
        
    Returns:
        float: The calculated learning rate for the epoch.
    """
    # Safety checks
    if total_epochs <= 0:
        raise ValueError("total_epochs must be greater than 0.")
    if epoch < 0:
        raise ValueError("epoch cannot be negative.")
        
    # Cap current epoch to total_epochs - 1 to prevent indexing issues at the final boundary
    t_cur = min(epoch, total_epochs)
    
    eta_max = lr0
    eta_min = lr0 * lrf
    
    # Calculate cosine factor
    cos_factor = math.cos((t_cur / total_epochs) * math.pi)
    
    # Compute learning rate
    lr = eta_min + 0.5 * (eta_max - eta_min) * (1.0 + cos_factor)
    return lr

def mutate_hyperparameters(
    parent_hyp: Dict[str, float],
    bounds: Dict[str, Tuple[float, float]],
    mutation_rate: float = 0.8,
    sigma: float = 0.1
) -> Dict[str, float]:
    """
    Applies a simple Genetic Algorithm mutation to a dictionary of hyperparameters.
    
    For each hyperparameter, with a probability of `mutation_rate`, we add a random
    mutation sampled from a Normal distribution:
        mutation ~ N(0, sigma * range)
    where range = upper_bound - lower_bound.
    
    Finally, the mutated values are clipped to their valid [lower, upper] boundaries.
    
    Args:
        parent_hyp (Dict[str, float]): Base (parent) hyperparameters (e.g., {'lr0': 0.01, ...}).
        bounds (Dict[str, Tuple[float, float]]): Valid range limits for each parameter.
        mutation_rate (float): Probability that any given hyperparameter is mutated.
        sigma (float): Standard deviation scale relative to the hyperparameter's range.
        
    Returns:
        Dict[str, float]: Mutated and clipped hyperparameters.
    """
    mutated_hyp = {}
    
    for key, parent_val in parent_hyp.items():
        if key not in bounds:
            # If no bounds are provided, do not mutate and keep original
            mutated_hyp[key] = parent_val
            continue
            
        lower, upper = bounds[key]
        param_range = upper - lower
        
        # Decide if this parameter should undergo mutation
        if random.random() < mutation_rate:
            # Generate mutation step using normal distribution scaled by parameter range and sigma
            mutation = random.gauss(0, sigma * param_range)
            mutated_val = parent_val + mutation
            
            # Clip to valid range
            clipped_val = max(lower, min(upper, mutated_val))
            mutated_hyp[key] = clipped_val
        else:
            # No mutation, keep parent value
            mutated_hyp[key] = parent_val
            
    return mutated_hyp

if __name__ == "__main__":
    print("=" * 60)
    print("🎓 EX50: Hyperparameter Tuning - Verification Tests")
    print("=" * 60)
    
    # --- 1. Test Cosine Annealing Learning Rate ---
    print("\n--- Test 1: Cosine Annealing LR Schedule ---")
    lr0_test = 0.01
    lrf_test = 0.01  # Final learning rate is 0.01 * 0.01 = 0.0001
    epochs_test = 10
    
    print(f"Initial LR (lr0): {lr0_test}")
    print(f"Final LR fraction (lrf): {lrf_test} -> Min LR: {lr0_test * lrf_test}")
    print(f"Total Epochs: {epochs_test}\n")
    
    for epoch in range(epochs_test + 1):
        lr = get_cosine_lr(epoch, epochs_test, lr0_test, lrf_test)
        print(f"  Epoch {epoch:2d}/{epochs_test}: Learning Rate = {lr:.6f}")
        
    # Check bounds correctness
    assert math.isclose(get_cosine_lr(0, epochs_test, lr0_test, lrf_test), lr0_test), "Epoch 0 must equal initial LR"
    assert math.isclose(get_cosine_lr(epochs_test, epochs_test, lr0_test, lrf_test), lr0_test * lrf_test), "Final epoch must equal min LR"
    print("\n✅ Cosine Annealing Math Verification Passed!")
    
    # --- 2. Test Genetic Algorithm Mutation ---
    print("\n--- Test 2: Genetic Algorithm Hyperparameter Mutation ---")
    
    # Base/parent hyperparameters
    parent_hyp = {
        'lr0': 0.01,
        'momentum': 0.937,
        'weight_decay': 0.0005,
        'mosaic': 1.0
    }
    
    # Define valid bounds for each parameter
    bounds = {
        'lr0': (1e-5, 0.1),
        'momentum': (0.6, 0.999),
        'weight_decay': (0.0, 0.01),
        'mosaic': (0.0, 1.0)
    }
    
    print("Base Parent Hyperparameters:")
    for k, v in parent_hyp.items():
        print(f"  {k:15s}: {v:.6f}  (Range: {bounds[k]})")
        
    # Seed for reproducibility of print statement checks
    random.seed(42)
    
    print("\nGenerating 5 Mutated Offspring:")
    for i in range(1, 6):
        offspring = mutate_hyperparameters(parent_hyp, bounds, mutation_rate=0.8, sigma=0.1)
        print(f"  Offspring {i}:")
        for k, v in offspring.items():
            diff = v - parent_hyp[k]
            diff_str = f"+{diff:.6f}" if diff >= 0 else f"{diff:.6f}"
            print(f"    {k:12s}: {v:.6f} (change: {diff_str})")
            
            # Assertion to make sure bounds are respected
            assert bounds[k][0] <= v <= bounds[k][1], f"Parameter {k} value {v} exceeds bounds!"
            
    print("\n✅ Genetic Algorithm Mutation & Clipping Verification Passed!")
    print("=" * 60)
