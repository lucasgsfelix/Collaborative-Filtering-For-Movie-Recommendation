"""

	First programming assignement of Recommender Systems class of 2020/2 - UFMG

    Our aim is to make a collaborative filtering for move recommendation

    Made by: Lucas Félix
    E-Mail: lucasgsfelix@gmail.com

"""

import sys

import model
import model_based
import metrics
import utils

import random

import time

def recommender_system(latent_factors, epochs, output_file):


    input_arguments = {"Historic Data": utils.read_table(sys.argv[1], ':', ','),
                       "Prediction Data": utils.read_table(sys.argv[2], ':')}


    with open("time_reports.csv", "a+") as time_report:

        model_based.non_negative_matrix_factorization(input_arguments, latent_factors, epochs, output_file)

        time_report.write('\t'.join([str(latent_factors), str(epochs), str(time.time() - start)]) + '\n')


if __name__ == '__main__':

    recommender_system(10, 20, "predictions.csv")
