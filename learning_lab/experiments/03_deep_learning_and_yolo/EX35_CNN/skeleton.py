import numpy as np

def calculate_output_shape(w_in, kernel_size, padding, stride):
    """
    Calculate the output dimension of a convolutional layer.
    """
    # TODO: Implement the output dimension formula: floor((w_in - kernel_size + 2 * padding) / stride) + 1
    return 0

def convolve2d_simple(img, kernel):
    """
    Perform a simple 2D convolution (stride=1, padding=0) on a 2D single channel image.
    """
    h_in, w_in = img.shape
    f_h, f_w = kernel.shape
    
    # TODO: Calculate output shape dimensions (without padding/stride)
    # Hint: h_out = h_in - f_h + 1
    h_out = 0
    w_out = 0
    
    output = np.zeros((h_out, w_out))
    
    # TODO: Slide the kernel and perform convolution
    # For i in range(h_out):
    #   For j in range(w_out):
    #     Extract region: img[i : i + f_h, j : j + f_w]
    #     output[i, j] = sum(region * kernel)
    
    return output

if __name__ == "__main__":
    img = np.array([
        [10, 10, 10, 0, 0, 0],
        [10, 10, 10, 0, 0, 0],
        [10, 10, 10, 0, 0, 0],
        [10, 10, 10, 0, 0, 0],
        [10, 10, 10, 0, 0, 0],
        [10, 10, 10, 0, 0, 0]
    ])
    
    kernel = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ])
    
    out = convolve2d_simple(img, kernel)
    shape_calc = calculate_output_shape(w_in=640, kernel_size=3, padding=1, stride=2)
    
    print("--- Training Results ---")
    if shape_calc != 0 or out.shape != (0, 0):
        print(f"Calculated Shape: {shape_calc} (expected: 320)")
        print(f"Convolved Map Shape: {out.shape} (expected: (4, 4))")
        print("Convolved Top-Left Corner Output:\n", out[:2, :2])
    else:
        print("Convolution calculations not implemented yet.")
