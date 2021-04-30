from comparison import visualization

import numpy as np 

def test_wrong_pixels():
    ground_truth = np.array([[1,2,3],[4,5,6],[7,8,9]])
    inference_mask = np.array([[1,2,4],[4,5,7],[7,8,10]])

    result = np.array([[False,False,True],[False,False,True],[False,False,True]])

    print("Wrong pixels test: ", end='')
    if(np.all(visualization.wrong_pixels(ground_truth, inference_mask) == result)):
        print('OK')
    else:
        print('ERROR')

def test_visualize_wrong_pixels():

    width = 500
    height = 500
    img = np.full((width, height, 3), (120,120,120))

    ground_truth = np.zeros((width, height, 1))

    ground_truth[40:60, 40:60] = 1

    inference_mask = np.ones((width, height, 1))

    visualization.visualize_wrong_pixels(img, ground_truth, inference_mask)