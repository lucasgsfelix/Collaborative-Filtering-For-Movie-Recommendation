"""

	Defining the operations for a model two work

"""
import math
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

def singular_value_decomposition_pp(data, latent_factors_size, epochs, error_metric):

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

    index_user = 0

    for epoch in range(epochs):

        for row in matrix_users_items:

            user, item = row[0], row[1]

            amount_itens = len(users_items[user])

            # diving all the values of a a array by the sqrt of the users amount of items
            ratings[user] = list(map(lambda value: value/math.sqrt(amount_itens), ratings[user]))

            # retriving all the values of a specific column
            column_array = retrieve_column(p_matrix, users[user])

            ratings[user] = algebric_operations.sum_two_arrays(ratings[user], column_array)

            predicted_rating = svd_prediction(p_matrix[users[user]], retrieve_column(q_matrix, items[item]))

            measured_error = error_metric(historic_rating_matrix[users[user]][items[item]], predicted_rating)

            print(measured_error)

            exit()


