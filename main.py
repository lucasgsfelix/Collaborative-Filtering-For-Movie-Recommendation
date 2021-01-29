"""

	First programming assignement of Recommender Systems class of 2020/2 - UFMG

    Our aim is to make a collaborative filtering for move recommendation

    Made by: Lucas Félix
    E-Mail: lucasgsfelix@gmail.com

"""

import sys

import model
import memory_based
import model_based_nmf
import model_based_svd
import metrics
import split_test
import utils

import random

import time

if __name__ == '__main__':


    input_arguments = {"Historic Data": utils.read_table(sys.argv[1], ':', ','),
                       "Prediction Data": utils.read_table(sys.argv[2], ':')}


    output_file = sys.argv[3]

    start = time.time()

    test = True

    input_arguments['Historic Data'] = random.sample(input_arguments['Historic Data'], int(len(input_arguments['Historic Data']) * 0.1) )

    if test:

        split_test.split_test(input_arguments)

    else:

        model_based_nmf.non_negative_matrix_factorization(input_arguments, 50, 20, output_file)


    end = time.time()

    print(end - start)

