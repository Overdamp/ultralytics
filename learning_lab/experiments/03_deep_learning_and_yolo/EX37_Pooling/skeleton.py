import numpy as np

def pool2d(img, pool_size=2, stride=2, mode='max'):
    """
    Perform a 2D pooling operation (Max or Average) on a single channel 2D image.
    """
    h_in, w_in = img.shape
    
    # TODO: Calculate output shape dimensions (height and width)
    # Hint: h_out = int((h_in - pool_size) / stride + 1)
    h_out = 0
    w_out = 0
    
    output = np.zeros((h_out, w_out))
    
    # TODO: Slide the pool window and apply Max or Average pooling
    # For i in range(h_out):
    #   For j in range(w_out):
    #     Extract region: img[i*stride : i*stride+pool_size, j*stride : j*stride+pool_size]
    #     If mode == 'max', calculate np.max(region)
    #     If mode == 'avg', calculate np.mean(region)
    for i in range(h_out):
        for j in range(w_out):
            pass
            
    return output

if __name__ == "__main__":
    img = np.array([
        [1.0, 3.0, 2.0, 9.0],
        [8.0, 4.0, 1.0, 5.0],
        [2.0, 7.0, 0.0, 3.0],
        [6.0, 5.0, 2.0, 6.0]
    ])
    
    max_out = pool2d(img, pool_size=2, stride=2, mode='max')
    avg_out = pool2d(img, pool_size=2, stride=2, mode='avg')
    
    print("--- Training Results ---")
    if max_out.shape != (0, 0):
        print("Max Pool Output:\n", max_out)
        print("Avg Pool Output:\n", avg_out)
    else:
        print("Pooling downsampling calculations not implemented yet.")
