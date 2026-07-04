# Back up or checkpoint this section of code before starting to modify the large file.
"""
EX49: YOLO Result Analysis - Solution
Professor's Note:
This is the complete pandas-based analysis implementation for diagnosing YOLO training logs.
We generate mock results.csv data representing two scenarios:
1. Normal convergence: Both training and validation losses decrease and stabilize.
2. Overfitting: Training loss decreases while validation loss starts rising in later epochs.

We then clean the data, find the best epoch by mAP, and perform automated overfitting detection
using linear regression (via numpy's polyfit) over the last N epochs.
"""

import pandas as pd
import numpy as np
import os

def load_and_clean_results(csv_path: str) -> pd.DataFrame:
    """
    Loads results.csv and cleans header names by removing whitespace.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Log file not found at {csv_path}")
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    return df

def find_best_epoch(df: pd.DataFrame, metric_col: str = 'metrics/mAP50-95(B)') -> dict:
    """
    Finds the epoch that achieved the maximum validation mAP.
    """
    if metric_col not in df.columns:
        raise ValueError(f"Metric column '{metric_col}' not found in DataFrame columns: {df.columns.tolist()}")
    
    best_row_idx = df[metric_col].idxmax()
    best_row = df.loc[best_row_idx]
    
    return {
        'best_epoch': int(best_row['epoch']),
        'best_mAP': float(best_row[metric_col])
    }

def check_overfitting(df: pd.DataFrame, window_size: int = 5, threshold: float = 0.0) -> dict:
    """
    Diagnoses overfitting by analyzing trends of training and validation losses over the last N epochs.
    
    We define:
      - Train Loss = train/box_loss + train/cls_loss + train/dfl_loss
      - Val Loss = val/box_loss + val/cls_loss + val/dfl_loss
      
    We fit a linear regression (first-order polynomial) to the last `window_size` epochs of loss data.
    If the slope of training loss is negative (decreasing) and the slope of validation loss is positive (increasing),
    we trigger an overfitting warning.
    """
    # 1. Compute total training and validation losses if individual components exist
    train_components = ['train/box_loss', 'train/cls_loss', 'train/dfl_loss']
    val_components = ['val/box_loss', 'val/cls_loss', 'val/dfl_loss']
    
    # Fallback to single loss if components are missing but a general loss column exists
    if all(col in df.columns for col in train_components):
        df['total_train_loss'] = df[train_components].sum(axis=1)
    elif 'train/loss' in df.columns:
        df['total_train_loss'] = df['train/loss']
    else:
        # If neither, try box loss as representative
        df['total_train_loss'] = df['train/box_loss']
        
    if all(col in df.columns for col in val_components):
        df['total_val_loss'] = df[val_components].sum(axis=1)
    elif 'val/loss' in df.columns:
        df['total_val_loss'] = df['val/loss']
    else:
        df['total_val_loss'] = df['val/box_loss']

    # 2. Check if we have enough epochs
    n_epochs = len(df)
    if n_epochs < window_size:
        return {
            'overfitting_detected': False,
            'train_loss_slope': 0.0,
            'val_loss_slope': 0.0,
            'message': f"Insufficient epochs ({n_epochs}) to analyze trend of window size {window_size}."
        }
        
    # 3. Extract the last `window_size` rows
    recent_df = df.tail(window_size)
    epochs = recent_df['epoch'].values
    train_losses = recent_df['total_train_loss'].values
    val_losses = recent_df['total_val_loss'].values
    
    # 4. Calculate trend slopes using numpy.polyfit
    # polyfit returns [slope, intercept] for degree 1
    train_slope, _ = np.polyfit(epochs, train_losses, 1)
    val_slope, _ = np.polyfit(epochs, val_losses, 1)
    
    # 5. Determine overfitting criteria
    # Training loss is decreasing (slope < threshold) AND Validation loss is increasing (slope > threshold)
    overfitting_detected = (train_slope < threshold) and (val_slope > -threshold) # slope > -threshold captures flat/increasing
    # More strictly: train_slope < threshold and val_slope > threshold
    strict_overfit = (train_slope < -1e-5) and (val_slope > 1e-5)
    
    if strict_overfit:
        message = (
            f"WARNING: Overfitting detected! Over the last {window_size} epochs, "
            f"Training Loss trend is DECREASING (slope = {train_slope:.5f}), but "
            f"Validation Loss trend is INCREASING (slope = {val_slope:.5f})."
        )
    elif train_slope < -1e-5 and val_slope > -1e-5:
        message = (
            f"ALERT: Potential plateau or early overfitting. Training Loss is decreasing (slope = {train_slope:.5f}), "
            f"but Validation Loss has stopped decreasing (slope = {val_slope:.5f})."
        )
    else:
        message = (
            f"Normal training behavior. Training Loss trend: {train_slope:.5f}, "
            f"Validation Loss trend: {val_slope:.5f} over the last {window_size} epochs."
        )
        
    return {
        'overfitting_detected': strict_overfit,
        'train_loss_slope': float(train_slope),
        'val_loss_slope': float(val_slope),
        'message': message
    }

def generate_mock_data(output_path: str, mode: str = 'normal', num_epochs: int = 50):
    """
    Generates a mock results.csv file simulating YOLOv8/YOLO11 outputs.
    
    Modes:
      - 'normal': train/val losses decrease, validation mAP climbs and plateaus.
      - 'overfit': train loss decreases, val loss decreases initially then rises after epoch 30.
    """
    epochs = np.arange(1, num_epochs + 1)
    time_per_epoch = 5.0 + np.random.normal(0, 0.2, num_epochs)
    
    # Training loss decreases exponentially
    train_box = 1.5 * np.exp(-epochs / 15) + 0.1 + np.random.normal(0, 0.01, num_epochs)
    train_cls = 2.0 * np.exp(-epochs / 12) + 0.05 + np.random.normal(0, 0.01, num_epochs)
    train_dfl = 1.0 * np.exp(-epochs / 18) + 0.02 + np.random.normal(0, 0.005, num_epochs)
    
    # Validation loss behavior depends on mode
    if mode == 'normal':
        # Decreases and plateaus
        val_box = 1.5 * np.exp(-epochs / 14) + 0.15 + np.random.normal(0, 0.02, num_epochs)
        val_cls = 2.0 * np.exp(-epochs / 11) + 0.1 + np.random.normal(0, 0.02, num_epochs)
        val_dfl = 1.0 * np.exp(-epochs / 16) + 0.04 + np.random.normal(0, 0.01, num_epochs)
        
        # mAP grows and plateaus
        mAP50 = 0.9 * (1.0 - np.exp(-epochs / 8)) + np.random.normal(0, 0.005, num_epochs)
        mAP50_95 = 0.7 * (1.0 - np.exp(-epochs / 9)) + np.random.normal(0, 0.005, num_epochs)
    else:
        # Overfitting mode: after epoch 25, validation loss begins to rise
        val_box = []
        val_cls = []
        val_dfl = []
        mAP50 = []
        mAP50_95 = []
        
        for e in epochs:
            # base exponential decay
            base_box = 1.5 * np.exp(-e / 14) + 0.15
            base_cls = 2.0 * np.exp(-e / 11) + 0.1
            base_dfl = 1.0 * np.exp(-e / 16) + 0.04
            
            # base mAP growth
            base_mAP50 = 0.9 * (1.0 - np.exp(-e / 8))
            base_mAP50_95 = 0.7 * (1.0 - np.exp(-e / 9))
            
            if e > 25:
                # Add overfitting drift
                drift = 0.03 * (e - 25)
                base_box += drift
                base_cls += drift * 1.5
                base_dfl += drift * 0.5
                
                # mAP degrades or plateaus
                base_mAP50 -= 0.005 * (e - 25)
                base_mAP50_95 -= 0.008 * (e - 25)
                
            val_box.append(base_box + np.random.normal(0, 0.01))
            val_cls.append(base_cls + np.random.normal(0, 0.01))
            val_dfl.append(base_dfl + np.random.normal(0, 0.005))
            mAP50.append(max(0.0, min(1.0, base_mAP50 + np.random.normal(0, 0.005))))
            mAP50_95.append(max(0.0, min(1.0, base_mAP50_95 + np.random.normal(0, 0.005))))
            
        val_box = np.array(val_box)
        val_cls = np.array(val_cls)
        val_dfl = np.array(val_dfl)
        mAP50 = np.array(mAP50)
        mAP50_95 = np.array(mAP50_95)
        
    precision = 0.85 * mAP50 + np.random.normal(0, 0.01, num_epochs)
    recall = 0.8 * mAP50 + np.random.normal(0, 0.01, num_epochs)
    
    # lr decay
    lr_pg0 = 0.01 * (1.0 - epochs / num_epochs)
    lr_pg1 = lr_pg0
    lr_pg2 = lr_pg0
    
    data = {
        'epoch': epochs,
        'time': time_per_epoch,
        'train/box_loss': train_box,
        'train/cls_loss': train_cls,
        'train/dfl_loss': train_dfl,
        'metrics/precision(B)': precision,
        'metrics/recall(B)': recall,
        'metrics/mAP50(B)': mAP50,
        'metrics/mAP50-95(B)': mAP50_95,
        'val/box_loss': val_box,
        'val/cls_loss': val_cls,
        'val/dfl_loss': val_dfl,
        'lr/pg0': lr_pg0,
        'lr/pg1': lr_pg1,
        'lr/pg2': lr_pg2
    }
    
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
    print(f"Mock results saved to: {output_path} ({mode} mode, {num_epochs} epochs)")

if __name__ == '__main__':
    # Define files
    normal_csv = 'results_mock_normal.csv'
    overfit_csv = 'results_mock_overfit.csv'
    
    # Generate mock CSVs
    generate_mock_data(normal_csv, mode='normal', num_epochs=40)
    generate_mock_data(overfit_csv, mode='overfit', num_epochs=40)
    
    print("\n--- Analysing Normal Convergence Scenario ---")
    df_normal = load_and_clean_results(normal_csv)
    best_normal = find_best_epoch(df_normal)
    diag_normal = check_overfitting(df_normal, window_size=5)
    print(f"Best Epoch: {best_normal['best_epoch']} with mAP50-95: {best_normal['best_mAP']:.4f}")
    print(f"Diagnostics: {diag_normal['message']}")
    
    print("\n--- Analysing Overfitting Scenario ---")
    df_overfit = load_and_clean_results(overfit_csv)
    best_overfit = find_best_epoch(df_overfit)
    diag_overfit = check_overfitting(df_overfit, window_size=5)
    print(f"Best Epoch: {best_overfit['best_epoch']} with mAP50-95: {best_overfit['best_mAP']:.4f}")
    print(f"Diagnostics: {diag_overfit['message']}")
    
    # Cleanup mock files after run
    if os.path.exists(normal_csv):
        os.remove(normal_csv)
    if os.path.exists(overfit_csv):
        os.remove(overfit_csv)
    print("\nCleanup completed.")
