import numpy as np

def pool2d(img, pool_size=2, stride=2, mode='max'):
    """
    Perform a 2D pooling operation (Max or Average) on a single channel 2D image.
    """
    h_in, w_in = img.shape
    
    h_out = int((h_in - pool_size) / stride + 1)
    w_out = int((w_in - pool_size) / stride + 1)
    
    output = np.zeros((h_out, w_out))
    
    for i in range(h_out):
        for j in range(w_out):
            r_start = i * stride
            r_end = r_start + pool_size
            c_start = j * stride
            c_end = c_start + pool_size
            
            region = img[r_start:r_end, c_start:c_end]
            
            if mode == 'max':
                output[i, j] = np.max(region)
            elif mode == 'avg':
                output[i, j] = np.mean(region)
                
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
    print("Max Pool Output:\n", max_out)
    print("Avg Pool Output:\n", avg_out)
