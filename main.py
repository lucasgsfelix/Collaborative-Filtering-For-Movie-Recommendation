"""

	First programming assignement of Recommender Systems class of 2020/2 - UFMG

    Our aim is to make a collaborative filtering for move recommendation

    Made by: Lucas Félix
    E-Mail: lucasgsfelix@gmail.com

"""

import sys

import model
import utils


import time

if __name__ == '__main__':


    input_arguments = {"Historic Data": utils.read_table(sys.argv[1], ':', ','),
                       "Prediction Data": utils.read_table(sys.argv[2], ':')}

    start = time.time()

    model.model_similarity_matrix(input_arguments['Historic Data'])

    end = time.time()

    print(end - start)


