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
    # TODO: Define a callback function log_map_metrics that accepts the 'validator' object
    # TODO: Extract the validation metrics dictionary from validator.metrics.results_dict
    # TODO: Retrieve the box mAP50 and mAP50-95 (with keys 'metrics/mAP50(B)' and 'metrics/mAP50-95(B)')
    # TODO: Print the metrics with clear formatting
    # TODO: Register the callback on the model using model.add_callback() for "on_val_end"
    # TODO: Return the modified model instance
    pass
