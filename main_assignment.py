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

def recommender_system(latent_factors, epochs, test=False, output_file=None):


    #input_arguments = {"Historic Data": utils.read_table(sys.argv[1], ':', ','),
    #                   "Prediction Data": utils.read_table(sys.argv[2], ':')}

    input_arguments = {"Historic Data": utils.read_table("Data/ratings.csv", ':', ','),
                       "Prediction Data": utils.read_table("Data/targets.csv", ':')}

    #output_file = sys.argv[3]

    start = time.time()


    if test:

        for test in range(0, 5):


            print("Iteration ", test, latent_factors, epochs)

            split_test.split_test(input_arguments, latent_factors, epochs)

    else:

        with open("Data/time_reports.csv", "a+") as time_report:


            model_based_nmf.non_negative_matrix_factorization(input_arguments, latent_factors, epochs, output_file)

            time_report.write('\t'.join([str(latent_factors), str(epochs), str(time.time() - start)]) + '\n')


    end = time.time()

    print(end - start)


if __name__ == '__main__':

    recommender_system(10, 10)
