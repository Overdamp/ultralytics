import numpy as np

def get_step_decay_lr(epoch, lr_initial=0.1, step_size=20, gamma=0.5):
    """
    Calculate learning rate using step decay.
    """
    # TODO: Calculate learning rate using step decay formula: lr_initial * (gamma ** (epoch // step_size))
    return 0.0

def get_cosine_annealing_lr(epoch, total_epochs=100, warmup_epochs=10, lr_max=0.1, lr_min=0.001):
    """
    Calculate learning rate using linear warmup followed by Cosine Annealing.
    """
    # TODO: Implement Linear Warmup if epoch < warmup_epochs
    # Hint: lr_min + (epoch / warmup_epochs) * (lr_max - lr_min)
    if epoch < warmup_epochs:
        pass
        
    # TODO: Implement Cosine Annealing if epoch >= warmup_epochs
    # Hint: lr_min + 0.5 * (lr_max - lr_min) * (1.0 + cos((t_cur / t_max) * pi))
    # where t_cur = epoch - warmup_epochs and t_max = total_epochs - warmup_epochs
    else:
        pass
        
    return 0.0

if __name__ == "__main__":
    print("--- Testing Step Decay Scheduler ---")
    step_val = get_step_decay_lr(25)
    if step_val != 0.0:
        print(f"Epoch 0:   lr = {get_step_decay_lr(0):.6f} (expected: 0.100000)")
        print(f"Epoch 25:  lr = {get_step_decay_lr(25):.6f} (expected: 0.050000)")
    else:
        print("Step decay not implemented yet.")
        
    print("\n--- Testing Cosine Annealing Scheduler ---")
    cosine_val = get_cosine_annealing_lr(10)
    if cosine_val != 0.0:
        print(f"Epoch 0:   lr = {get_cosine_annealing_lr(0):.6f} (expected: 0.001000)")
        print(f"Epoch 5:   lr = {get_cosine_annealing_lr(5):.6f} (expected: 0.050500)")
        print(f"Epoch 10:  lr = {get_cosine_annealing_lr(10):.6f} (expected: 0.100000)")
        print(f"Epoch 55:  lr = {get_cosine_annealing_lr(55):.6f} (expected: 0.050500)")
        print(f"Epoch 100: lr = {get_cosine_annealing_lr(100):.6f} (expected: 0.001000)")
    else:
        print("Cosine annealing not implemented yet.")
