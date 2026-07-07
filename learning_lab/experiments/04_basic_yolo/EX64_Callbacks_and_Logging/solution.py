from ultralytics import YOLO

def register_custom_metric_callback(model: YOLO) -> YOLO:
    """
    Defines a custom callback for the 'on_val_end' event that extracts and logs
    validation metrics (mAP50 and mAP50-95), registers the callback on the YOLO model instance,
    and returns the modified model.

    Args:
        model (YOLO): The Ultralytics YOLO model instance.

    Returns:
        YOLO: The modified YOLO model instance with the registered callback.
    """
    def log_map_metrics(validator):
        """
        Callback function executed at the end of validation.
        Extracts and logs mAP50 and mAP50-95 from the validator's metrics.
        """
        metrics = validator.metrics
        results = metrics.results_dict
        
        # Access detection metrics (with '(B)' indicating bounding box metrics)
        map50 = results.get("metrics/mAP50(B)", 0.0)
        map50_95 = results.get("metrics/mAP50-95(B)", 0.0)
        
        print("\n=========================================")
        print("🎓 Professor Monitor - Validation End")
        print(f"--> mAP50 (Box): {map50:.6f}")
        print(f"--> mAP50-95 (Box): {map50_95:.6f}")
        print("=========================================\n")

    # Register callback on the model
    model.add_callback("on_val_end", log_map_metrics)
    
    return model
