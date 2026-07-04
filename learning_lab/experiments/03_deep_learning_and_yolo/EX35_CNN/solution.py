import numpy as np

def calculate_output_shape(w_in, kernel_size, padding, stride):
    """
    Calculate the output dimension of a convolutional layer.
    """
    return int(np.floor((w_in - kernel_size + 2 * padding) / stride) + 1)

def convolve2d_simple(img, kernel):
    """
    Perform a simple 2D convolution (stride=1, padding=0) on a 2D single channel image.
    """
    h_in, w_in = img.shape
    f_h, f_w = kernel.shape
    
    h_out = h_in - f_h + 1
    w_out = w_in - f_w + 1
    
    output = np.zeros((h_out, w_out))
    
    for i in range(h_out):
        for j in range(w_out):
            region = img[i : i + f_h, j : j + f_w]
            output[i, j] = np.sum(region * kernel)
            
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
    print(f"Calculated Shape: {shape_calc} (expected: 320)")
    print(f"Convolved Map Shape: {out.shape} (expected: (4, 4))")
    print("Convolved Top-Left Corner Output:\n", out[:2, :2])
