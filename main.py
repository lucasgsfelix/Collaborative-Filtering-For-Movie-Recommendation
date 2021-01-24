"""

	First programming assignement of Recommender Systems class of 2020/2 - UFMG

    Our aim is to make a collaborative filtering for move recommendation

    Made by: Lucas Félix
    E-Mail: lucasgsfelix@gmail.com

"""

import sys

import model
import memory_based
import model_based
import metrics
import utils


import time

if __name__ == '__main__':


    input_arguments = {"Historic Data": utils.read_table(sys.argv[1], ':', ','),
                       "Prediction Data": utils.read_table(sys.argv[2], ':')}

    start = time.time()

    model_based.singular_value_decomposition_pp(input_arguments, 10, 10, metrics.root_mean_squared)

    #memory_based.measure_ratings_by_nearest_neighbors(input_arguments, 'items')

    end = time.time()

    print(end - start)


