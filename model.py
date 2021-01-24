"""

	Defining the operations for a model two work

"""
import math
import random
import algebric_operations
import data_treatment
import utils

def calculate_first_estimation(users, users_items, latent_factors_size, y_matrix, items):

    # first estimation of the ratings
    estimation = {}

    for user in users.keys():
        # array of zeros
        zero_array = [0] * latent_factors_size

        for item in users_items[user]: # items consumed by the user

            zero_array = algebric_operations.sum_two_arrays(zero_array, y_matrix[items[item]])

        estimation[user] = zero_array

    return estimation

def retrieve_column(matrix, column):

    column_array = []

    for value in range(len(matrix)):

        column_array.append(matrix[value][column])

    return column_array

def svd_prediction(p_matrix, q_matrix):

    return sum(list(map(lambda x, y: x*y, p_matrix, q_matrix)))


def generate_historic_data_matrix(historic_data, modeling, users, items):
    """
        Modeling the matrix of historical data

        Modeling:

            Define if the matrix generate will be a item x item or user x user modeling


    """

    # making a matrix of zeros
    if modeling == 'items':

        matrix = [[0] * len(users) for row in range(0, len(items))]

    else:

        matrix = [[0] * len(items) for row in range(0, len(users))]


    for row in historic_data:

        user = users[row[0]]
        item = items[row[1]]

        # rating given by the user
        rating = int(row[2])

        if modeling == 'items':

            matrix[item][user] = rating

        else:

            matrix[user][item] = rating

    return matrix


def _update_p_matrix(p_matrix, q_matrix, user_index, item_index, error, lambda_value=0.1, gamma_value=0.01):

    for row in range(len(p_matrix)):

        p_matrix[row][user_index] += gamma_value * (error * q_matrix[row][item_index] - lambda_value * p_matrix[row][user_index])

    return p_matrix


def _update_q_matrix(q_matrix, p_matrix, user_index, item_index, user, amount_items, lambda_value=0.05, gamma_value=0.01):

    for row in range(len(q_matrix)):

        q_matrix[row][item_index] += gamma_value * ((p_matrix[row][user_index] + 1)/math.sqrt(amount_items) * ratings[user][row_index]) - lambda_value * q_matrix[row][item_index]

    return q_matrix

def _update_y_matrix(y_matrix, q_matrix, user_items, user, items, amount_items, lambda_value=0.1, gamma_value=0.01):

    for row, item in enumerate(users_items[item]):

        for column in range(len(y_matrix[items[item]])):

            y_matrix[items[item]][column] += gamma_value * (error * 1/math.sqrt(amount_items) * q_matrix[column][items[item]] - lambda_value * y_matrix[items[item], column])

    return y_matrix


def _update_residual_items(residual_items, item_index, error, gamma_value=0.01, lambda_value=0.05):


    residual_items[item_index] += gamma_value * (error - lambda_value * residual_items[item_index])


def _update_residual_users(residual_users, user_index, error, gamma_value=0.01, lambda_value=0.05):


    residual_users[user_index] += gamma_value * (error - lambda_value * residual_users[user_index])


def singular_value_decomposition_pp(data, latent_factors_size, epochs, error_metric):
    """

        Based on the code available in:

            https://github.com/cheungdaven/recommendation


    """

    users_items, users, items = data_treatment.retrieve_guide_features(data['Historic Data'])

    matrix_users_items = data_treatment.mount_matrix_user_item(users_items)

    # a matrix users x items
    historic_rating_matrix = generate_historic_data_matrix(data['Historic Data'], 'users', users, items)

    # users latent matrix
    p_matrix = algebric_operations.generate_random_matrix(latent_factors_size, len(users))

    # itens latent matrix
    q_matrix = algebric_operations.generate_random_matrix(latent_factors_size, len(items))

    # prediction matrix
    y_matrix = algebric_operations.generate_random_matrix(len(items), latent_factors_size)

    ratings = calculate_first_estimation(users, users_items, latent_factors_size, y_matrix, items)

    residual_items = [random.uniform(0, 1) for item in range(0, len(items))]

    residual_users = [random.uniform(0, 1) for user in range(0, len(users))]

    index_user = 0

    for epoch in range(epochs):

        for row in matrix_users_items:

            user, item = row[0], row[1]

            user_index, item_index = users[user], items[item]

            amount_itens = len(users_items[user])

            # diving all the values of a a array by the sqrt of the users amount of items
            ratings[user] = list(map(lambda value: value/math.sqrt(amount_itens), ratings[user]))

            # retriving all the values of a specific column
            column_array = retrieve_column(p_matrix, users[user])

            ratings[user] = algebric_operations.sum_two_arrays(ratings[user], column_array)

            predicted_rating = svd_prediction(p_matrix[user_index], retrieve_column(q_matrix, items[item]))

            measured_error = error_metric(historic_rating_matrix[users[user]][item_index], predicted_rating)

            p_matrix = _update_p_matrix(p_matrix, q_matrix, user_index, item_index, measured_error)

            q_matrix = _update_q_matrix(q_matrix, p_matrix, user_index, item_index, user, amount_items)

            y_matrix = _update_y_matrix(y_matrix, q_matrix, user_items, user, items, amount_items)

            residual_items = _update_residual_items(residual_items, item_index, measured_error)

            residual_users = _update_residual_users(residual_users, user_index, measured_error)
