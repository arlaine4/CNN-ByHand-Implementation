import argparse
import numpy as np
import random
import sys

class Options:
    def __init__(self, stride, padding, dimension, image, filtre):
        self.stride = stride
        self.padding = padding
        self.dimension = dimension
        self.image = image
        self.filter = filtre

def init_conv_layer(image, filtre, stride=0, padding=0, dimension=2):
    options = Options(stride, padding, dimension, image, filtre)
    image_matrix, filter_matrix = generate_matricies(options)
    print("\nGenerated image : \n\n", image_matrix)
    print("\nGenerated filter : \n\n", filter_matrix)
    result_matrix = convolution(image_matrix, filter_matrix, options)
    return result_matrix

def generate_matricies(options):
    if not options.dimension or options.dimension == 2:
        if options.image > 0 and options.filter > 0:
            base_matrix = np.zeros((options.image, options.image))
            filter_matrix = np.zeros((options.filter, options.filter))
        else:
            sys.exit("You can't give negative dimensions for image matrix, neither for filter matrix.")
        #----------------------------------------------------------------------------------------#
        #                           Artificial filter and image creation                         #
        for i in range(base_matrix.shape[0]):
            for j in range(base_matrix.shape[1]):
                base_matrix[i][j] = random.randint(0 - options.image*2, 0 + options.image**2)
                #base_matrix[i][j] = random.randint(-2, 5)
        for i in range(filter_matrix.shape[0]):
            for j in range(filter_matrix.shape[1]):
                filter_matrix[i][j] = random.randint(0 - options.filter*2, 0 + options.filter**2)
                #filter_matrix[i][j] = random.randint(-2, 5)
        #                                                                                        #
        #----------------------------------------------------------------------------------------#
        if options.padding:
            base_matrix = np.pad(base_matrix, options.padding)
        return base_matrix, filter_matrix
    elif options.dimension > 3 or options.dimensions < 2:
        sys.exit("I can't handle images of dimension {}".format(options.dimension))


def convolution(image, filtre, options):
    """
        params:
            image   -> n*n*n_c matrix -> 2 or 3d depending on the params inside options
            filtre  -> f*f*f_c matric -> 2 or 3d depending on the params inside options
            options -> list of options for the convolution layer
    """

    #---------------------------------------------------------------------------------------------------#
    #                                   Basic padding param handling                                    #
    if options.padding and options.padding < 0:
        sys.exit("Can't deal with a negative padding.")
    if options.stride and (options.stride < 0 or options.stride >= image.shape[0] - 1):
        sys.exit("Invalid stride param, it's either too high or too low.")
    #                                                                                                   #
    #---------------------------------------------------------------------------------------------------#

    # Adjusting padding and stride values if they are NaN
    options.padding = 0 if not options.padding else options.padding
    options.stride = 1 if not options.stride else options.stride

    # Result_matrix dimension calcul
    result_dim = int(((image.shape[0] + 2 * (options.padding) - filtre.shape[0]) / options.stride) + 1)

    # ReAdjusting stride
    options.stride = 0 if options.stride == 1 else options.stride

    if not options.dimension or options.dimension == 2:
        result_matrix = np.zeros((result_dim, result_dim))

    # Main loop filling the result matrix
    for i in range(result_dim):
        for j in range(result_dim):
            result_matrix[i][j] = proceed_sum_matrix(i, j, options.stride, image, filtre)
    print("\nResult matrix : \n\n", result_matrix)
    return result_matrix

def proceed_sum_matrix(y, x, stride, image, filtre):
    #-----------------------------------------#
    #       Adjusting x and y for stride      #
    case_result = 0
    if y > 0: 
        y += stride - 1 if y == 1 else stride
    if (y == 0 and x > 0) or (y > 0 and x > 0):
        x += stride - 1 if x == 1 else stride
    #-----------------------------------------#

    for i, i_f in zip(range(y, y + filtre.shape[0]), range(filtre.shape[0])):
        for j, j_f in zip(range(x, x + filtre.shape[0]), range(filtre.shape[0])):
            case_result += image[i][j] * filtre[i_f][j_f]

    return case_result

