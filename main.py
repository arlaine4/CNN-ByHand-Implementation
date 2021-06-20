import numpy as np
import convolution_layer as conv

if __name__ == "__main__":
    result_matrix = conv.init_conv_layer(14, 4, stride=2)
    print("\nResult matrix : \n\n", result_matrix)
    result_matrix = conv.init_conv_layer(-1, 3,true_image=result_matrix) 
    print("\nResult matrix : \n\n", result_matrix)
    result_matrix = conv.init_conv_layer(-1, 3,true_image=result_matrix) 
    print("\nResult matrix : \n\n", result_matrix)
    
