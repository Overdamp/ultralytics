def find_early_stopping(val_losses, patience=5):
    """
    Scans validation losses to find the optimal stopping epoch.
    Returns:
    - best_epoch (int): 1-indexed epoch with the minimum validation loss
    - stopped_epoch (int): 1-indexed epoch where training is stopped
    """
    # TODO: Initialize tracking parameters: best_loss = inf, best_epoch = 0, no_improvement_count = 0
    best_loss = float('inf')
    best_epoch = 0
    
    # TODO: Loop over val_losses, updating best loss/epoch and tracking no-improvement epochs
    # Hint: If current loss < best_loss, reset no_improvement_count and save the new best_loss.
    # Otherwise, increment no_improvement_count.
    # If no_improvement_count >= patience, stop training early and return.
    for idx, loss in enumerate(val_losses):
        pass
        
    return best_epoch, len(val_losses)

if __name__ == "__main__":
    val_losses = [2.5, 1.8, 1.4, 1.2, 1.1, 1.15, 1.2, 1.3, 1.4, 1.5]
    
    best_ep, stop_ep = find_early_stopping(val_losses, patience=3)
    
    print("--- Training Results ---")
    if best_ep != 0:
        print(f"Optimal Epoch:  {best_ep} (expected: 5)")
        print(f"Stopped Epoch:  {stop_ep} (expected: 8)")
    else:
        print("Early stopping detection logic not implemented yet.")
