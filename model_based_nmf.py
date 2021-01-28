"""

    Collaborative Filtering: Modeling Methods

"""

import math
import random

import algebric_operations
import data_treatment
import model
import utils



def nmf_prediction(p_matrix, q_matrix):

    return sum(list(map(lambda x, y: x*y, p_matrix, q_matrix)))


def _update_matrixes(p_matrix, q_matrix, user_index, item_index, error, lambda_value=0.1, gamma_value=0.01):

    for row in range(len(p_matrix)):

        p_matrix[row][user_index] += gamma_value * (error * q_matrix[row][item_index] - lambda_value * p_matrix[row][user_index])

    return p_matrix


def measure_rmse(matrix_users_items, historic_rating_matrix, p_matrix, q_matrix, users, items):


    total_error = 0

    for row in matrix_users_items:

        user, item = row[0], row[1]

        user_index, item_index = users[user], items[item]

        total_error += (historic_rating_matrix[user_index][item_index] - nmf_prediction(retrieve_column(p_matrix, user_index), retrieve_column(q_matrix, item_index))) ** 2

    return math.sqrt((total_error)/len(matrix_users_items))

def make_prediction(prediction_data, p_matrix, q_matrix, ratings_mean, users, items, users_mean, items_mean):

    predictions = []

    for row in prediction_data:

        user, item = row[0], row[1]

        if user in users.keys() and item not in items.keys():

            prediction = users_mean[user]

        elif item in items.keys() and user not in users.keys():

            prediction = items_mean[item]

        if user not in users.keys() or item not in items.keys():

            prediction = ratings_mean

        else:

            user_index, item_index = users[user], items[item]

            prediction = nmf_prediction(retrieve_column(p_matrix, user_index), retrieve_column(q_matrix, item_index))

        predictions.append(prediction)  

    return predictions

def retrieve_column(matrix, column):

    column_array = []

    for value in range(len(matrix)):

        column_array.append(matrix[value][column])

    return column_array

def non_negative_matrix_factorization(data, latent_factors_size, epochs):
    """

        Based on the code available in:

            https://github.com/cheungdaven/recommendation/blob/master/recSysNMF.py

        We also use as guide:


            https://blog.acolyer.org/2019/02/18/the-why-and-how-of-nonnegative-matrix-factorization/
            Class 08 - Collaborative Filtering: Factorization Matrix

    """
    random.seed()

    users_items, users, items, users_ratings, items_ratings = data_treatment.retrieve_guide_features(data['Historic Data'])

    tokens = utils.retrieve_unique_tokens(data['Prediction Data'])

    matrix_users_items = data_treatment.mount_matrix_user_item(users_items)

    ratings_mean = utils.measure_average_rating(data['Historic Data'])

    # a matrix users x items
    historic_rating_matrix = model.generate_historic_data_matrix(data['Historic Data'], 'users', users, items, ratings_mean)

    # users latent matrix
    p_matrix = algebric_operations.generate_random_matrix(latent_factors_size, len(users))

    # itens latent matrix
    q_matrix = algebric_operations.generate_random_matrix(latent_factors_size, len(items))

    for epoch in range(epochs):

        for row in matrix_users_items:

            user, item = row[0], row[1]

            user_index, item_index = users[user], items[item]

            error =  historic_rating_matrix[user_index][item_index] - nmf_prediction(retrieve_column(p_matrix, user_index), retrieve_column(q_matrix, item_index))

            p_matrix = _update_matrixes(p_matrix, q_matrix, user_index, item_index, error)

            q_matrix = _update_matrixes(q_matrix, p_matrix, item_index, user_index, error)

        print(measure_rmse(matrix_users_items, historic_rating_matrix, p_matrix, q_matrix, users, items))

    predictions = make_prediction(data['Prediction Data'], p_matrix, q_matrix, ratings_mean, users, items, users_ratings, items_ratings)

    for index, prediction in enumerate(predictions):

        data['Prediction Data'][index].append(str(prediction))

    data['Prediction Data'].insert(0, ['UserId', 'ItemId', 'Prediction'])

    utils.write_table(data['Prediction Data'], "Outputs/predictions.txt")

