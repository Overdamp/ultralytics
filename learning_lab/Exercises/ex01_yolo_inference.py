# ex01_yolo_inference.py
# Exercise 1: Writing code to read an image and run YOLO inference
# Fill in the # TODO lines below.

# TODO: Import the YOLO class from the correct ultralytics package
# Hint: from ultralytics import ...


def run_inference():
    # TODO: Load the pre-trained weights file 'yolo26n.pt'
    # Hint: model = YOLO(...)
    model = None

    # TODO: Run inference on the local image 'bus.jpg'
    # Hint: results = model(...)
    results = None

    # Verify if inference ran successfully
    if results is None:
        print("Model inference not implemented yet!")
        return

    # TODO: Loop through the results and print out predicted class labels and bounding box coordinates
    # Hint: Loop through results, then loop through result.boxes.
    # Print the coordinates (box.xyxy), confidence (box.conf), and class name.
    print("\n--- Detection Results ---")
    for result in results:
        # Loop through each bounding box
        pass


if __name__ == "__main__":
    run_inference()
