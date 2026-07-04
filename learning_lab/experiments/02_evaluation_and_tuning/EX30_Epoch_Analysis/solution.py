def find_early_stopping(val_losses, patience=5):
    """
    Scans validation losses to find the optimal stopping epoch.
    Returns:
    - best_epoch (int): 1-indexed epoch with the minimum validation loss
    - stopped_epoch (int): 1-indexed epoch where training is stopped
    """
    best_loss = float('inf')
    best_epoch = 0
    no_improvement_count = 0
    
    for idx, loss in enumerate(val_losses):
        epoch = idx + 1
        if loss < best_loss:
            best_loss = loss
            best_epoch = epoch
            no_improvement_count = 0
        else:
            no_improvement_count += 1
            
        if no_improvement_count >= patience:
            return best_epoch, epoch
            
    return best_epoch, len(val_losses)

if __name__ == "__main__":
    # Simulated validation losses: minimum is at index 4 (value 1.1), which is epoch 5
    val_losses = [2.5, 1.8, 1.4, 1.2, 1.1, 1.15, 1.2, 1.3, 1.4, 1.5]
    
    best_ep, stop_ep = find_early_stopping(val_losses, patience=3)
    
    print("--- Training Results ---")
    print(f"Optimal Epoch:  {best_ep} (expected: 5)")
    print(f"Stopped Epoch:  {stop_ep} (expected: 8)")
