def scale_learning_rate(lr_base, batch_base, batch_target):
    """
    Calculate the new learning rate based on the Linear Scaling Rule.
    """
    # TODO: Implement Linear Scaling Rule: lr_base * (batch_target / batch_base)
    return 0.0

if __name__ == "__main__":
    lr_base = 0.01
    batch_base = 16
    
    scaled_val = scale_learning_rate(lr_base, batch_base, 32)
    
    print("--- Testing Linear Scaling Rule ---")
    if scaled_val != 0.0:
        print(f"Batch 16 -> 32: lr = {scale_learning_rate(lr_base, batch_base, 32):.4f} (expected: 0.0200)")
        print(f"Batch 16 -> 64: lr = {scale_learning_rate(lr_base, batch_base, 64):.4f} (expected: 0.0400)")
        print(f"Batch 16 -> 8:  lr = {scale_learning_rate(lr_base, batch_base, 8):.4f} (expected: 0.0050)")
    else:
        print("Linear scaling rule not implemented yet.")
