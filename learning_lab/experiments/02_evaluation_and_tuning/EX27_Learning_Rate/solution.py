import numpy as np

def get_step_decay_lr(epoch, lr_initial=0.1, step_size=20, gamma=0.5):
    """
    Calculate learning rate using step decay.
    """
    return lr_initial * (gamma ** (epoch // step_size))

def get_cosine_annealing_lr(epoch, total_epochs=100, warmup_epochs=10, lr_max=0.1, lr_min=0.001):
    """
    Calculate learning rate using linear warmup followed by Cosine Annealing.
    """
    if epoch < warmup_epochs:
        # Linear Warmup from lr_min to lr_max
        return lr_min + (epoch / warmup_epochs) * (lr_max - lr_min)
    else:
        # Cosine Annealing from lr_max to lr_min
        t_cur = epoch - warmup_epochs
        t_max = total_epochs - warmup_epochs
        return lr_min + 0.5 * (lr_max - lr_min) * (1.0 + np.cos((t_cur / t_max) * np.pi))

if __name__ == "__main__":
    print("--- Testing Step Decay Scheduler ---")
    print(f"Epoch 0:   lr = {get_step_decay_lr(0):.6f} (expected: 0.100000)")
    print(f"Epoch 25:  lr = {get_step_decay_lr(25):.6f} (expected: 0.050000)")
    
    print("\n--- Testing Cosine Annealing Scheduler ---")
    print(f"Epoch 0:   lr = {get_cosine_annealing_lr(0):.6f} (expected: 0.001000)")
    print(f"Epoch 5:   lr = {get_cosine_annealing_lr(5):.6f} (expected: 0.050500)")
    print(f"Epoch 10:  lr = {get_cosine_annealing_lr(10):.6f} (expected: 0.100000)")
    print(f"Epoch 55:  lr = {get_cosine_annealing_lr(55):.6f} (expected: 0.050500)")
    print(f"Epoch 100: lr = {get_cosine_annealing_lr(100):.6f} (expected: 0.001000)")
