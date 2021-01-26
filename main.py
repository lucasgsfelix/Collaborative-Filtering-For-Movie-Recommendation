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

import random

import time

if __name__ == '__main__':


    input_arguments = {"Historic Data": utils.read_table(sys.argv[1], ':', ','),
                       "Prediction Data": utils.read_table(sys.argv[2], ':')}

    start = time.time()

    #input_arguments['Historic Data'] = random.sample(input_arguments['Historic Data'], int(len(input_arguments['Historic Data']) * 0.1))

    model_based.singular_value_decomposition_pp(input_arguments, 10, 100)

    #memory_based.measure_ratings_by_nearest_neighbors(input_arguments, 'items')

    end = time.time()

    print(end - start)

