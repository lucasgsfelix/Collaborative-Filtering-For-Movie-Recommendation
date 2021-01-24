"""

    Collaborative Filtering: Modeling Methods

"""

import math
import random

import algebric_operations
import data_treatment
import model
import utils

def _update_p_matrix(p_matrix, q_matrix, user_index, item_index, error, lambda_value=0.1, gamma_value=0.01):

    for row in range(len(p_matrix)):

        p_matrix[row][user_index] += gamma_value * (error * q_matrix[row][item_index] - lambda_value * p_matrix[row][user_index])

    return p_matrix

def _update_q_matrix(q_matrix, p_matrix, user_index, item_index, user, amount_items, ratings, lambda_value=0.05, gamma_value=0.01):

    for row in range(len(q_matrix)):

        q_matrix[row][item_index] += gamma_value * ((p_matrix[row][user_index] + 1)/math.sqrt(amount_items) * ratings[user][row]) - lambda_value * q_matrix[row][item_index]

    return q_matrix

def _update_y_matrix(y_matrix, q_matrix, users_items, user, item_index, amount_items, error, lambda_value=0.1, gamma_value=0.01):

    for row, item in enumerate(users_items[user]):

        for column in range(len(y_matrix[item_index])):

            y_matrix[item_index][column] += gamma_value * (error * 1/math.sqrt(amount_items) * q_matrix[column][item_index] - lambda_value * y_matrix[item_index][column])

    return y_matrix

def _update_residual_items(residual_items, item_index, error, gamma_value=0.01, lambda_value=0.05):


    residual_items[item_index] += gamma_value * (error - lambda_value * residual_items[item_index])

    return residual_items

def _update_residual_users(residual_users, user_index, error, gamma_value=0.01, lambda_value=0.05):


    residual_users[user_index] += gamma_value * (error - lambda_value * residual_users[user_index])

    return residual_users

def measure_average_rating(data):


    ratings_sum = 0

    for row in data:

        ratings_sum += float(row[2])

    return ratings_sum/len(data)

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

def rmse(I, R, item_by_users,average, Q, P, Y, B_U, B_I):
    users, items = R.nonzero()
    sum = 0
    for u, i in zip(users, items):  # 75000 times
        n_u = len(users[users == u])
        if (R[u, i] > 5 or R[u, i] < 0):
            print("R[" + str(u) + "," + str(i) + "]=" + R[u, i])
        # print(n_u)
        pPlusY = np.zeros(numFactors)
        for j in item_by_users[u]:
            pPlusY = np.add(pPlusY, Y[j, :])
        # print(pPlusY)
        pPlusY = np.add(pPlusY / np.sqrt(n_u), P[:, u])
        error = R[u, i] - (average + B_U[u] + B_I[i] + prediction(pPlusY, Q[:, i]))
        sum += error**2
    return np.sqrt(sum/len(R[R > 0]))

def svd_rmse(historic_rating_matrix, matrix_users_items, users, items, ratings_mean, residual_users, residual_items, ratings, q_matrix):

    total_error = 0

    for row in matrix_users_items:

        user, item = row[0], row[1]        

        user_index, item_index = users[user], items[item]

        prediction = (ratings_mean + residual_users[user_index] + residual_items[item_index] + svd_prediction(ratings[user], retrieve_column(q_matrix, item_index)))

        total_error += (historic_rating_matrix[user_index][item_index] - prediction) ** 2

    return math.sqrt(total_error/len(matrix_users_items))


def make_prediction(prediction_data, ratings, ratings_mean, users, items, q_matrix, residual_users, residual_items):

    predictions = []

    for row in prediction_data:

        user, item = row[0], row[1]

        if user not in users.keys() or item not in items.keys():

            prediction = ratings_mean

        else:

            user_index, item_index = users[user], items[item]

            prediction = (ratings_mean + residual_users[user_index] + residual_items[item_index] + svd_prediction(ratings[user], retrieve_column(q_matrix, item_index)))

        predictions.append(prediction)

    return predictions


def singular_value_decomposition_pp(data, latent_factors_size, epochs, error_metric):
    """

        Based on the code available in:

            https://github.com/cheungdaven/recommendation


    """
    random.seed()

    users_items, users, items = data_treatment.retrieve_guide_features(data['Historic Data'])

    matrix_users_items = data_treatment.mount_matrix_user_item(users_items)

    ratings_mean = measure_average_rating(data['Historic Data'])

    # a matrix users x items
    historic_rating_matrix = model.generate_historic_data_matrix(data['Historic Data'], 'users', users, items, ratings_mean)

    # users latent matrix
    p_matrix = algebric_operations.generate_random_matrix(latent_factors_size, len(users))

    # itens latent matrix
    q_matrix = algebric_operations.generate_random_matrix(latent_factors_size, len(items))

    # prediction matrix
    y_matrix = algebric_operations.generate_random_matrix(len(items), latent_factors_size)

    ratings = calculate_first_estimation(users, users_items, latent_factors_size, y_matrix, items)

    residual_items = [random.uniform(0, 1) for item in range(0, len(items))]

    residual_users = [random.uniform(0, 1) for user in range(0, len(users))]

    for epoch in range(epochs):

        for row in matrix_users_items:

            user, item = row[0], row[1]

            user_index, item_index = users[user], items[item]

            amount_items = len(users_items[user])

            # diving all the values of a a array by the sqrt of the users amount of items
            ratings[user] = list(map(lambda value: value/math.sqrt(amount_items), ratings[user]))

            # retriving all the values of a specific column
            column_array = retrieve_column(p_matrix, users[user])

            ratings[user] = algebric_operations.sum_two_arrays(ratings[user], column_array)

            predicted_rating = ratings_mean + residual_items[item_index] + residual_users[user_index] + svd_prediction(ratings[user], retrieve_column(q_matrix, item_index))

            measured_error = historic_rating_matrix[user_index][item_index] - predicted_rating # error_metric(historic_rating_matrix[users[user]][item_index], predicted_rating)

            p_matrix = _update_p_matrix(p_matrix, q_matrix, user_index, item_index, measured_error)

            q_matrix = _update_q_matrix(q_matrix, p_matrix, user_index, item_index, user, amount_items, ratings)

            #y_matrix = _update_y_matrix(y_matrix, q_matrix, users_items, user, item_index, amount_items, measured_error)

            residual_items = _update_residual_items(residual_items, item_index, measured_error)

            residual_users = _update_residual_users(residual_users, user_index, measured_error)

        print(svd_rmse(historic_rating_matrix, matrix_users_items, users, items, ratings_mean, residual_users, residual_items, ratings, q_matrix))

    predictions = make_prediction(data['Prediction Data'], ratings, ratings_mean, users, items, q_matrix, residual_users, residual_items)

    for index, prediction in enumerate(predictions):

        data['Prediction Data'][index].append(str(prediction))

    data['Prediction Data'].insert(0, ['UserId', 'ItemId', 'Prediction'])

    utils.write_table(data['Prediction Data'], "Outputs/predictions.txt")

