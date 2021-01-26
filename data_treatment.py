"""

    Data Treatment and Utility Functions for data Identification

"""

import utils

def define_index(tokens):

    return {token: index for index, token in enumerate(tokens)}


def retrieve_guide_features(historic_data):

    tokens = utils.retrieve_unique_tokens(historic_data)

    users_items = {user: [] for user in tokens['users']}

    items_ratings = dict(zip(tokens['items'], [[]] * len(tokens['items'])))
    users_ratings = dict(zip(tokens['users'], [[]] * len(tokens['users'])))

    for row in historic_data:

        users_items[row[0]].append(row[1])

        items_ratings[row[1]].append(float(row[2]))

        users_ratings[row[0]].append(float(row[2]))


    '''for user in users_ratings.keys():

        users_ratings[user] = sum(users_ratings[user])/len(users_ratings[user])

    for item in items_ratings.keys():

        items_ratings[item] = sum(items_ratings[item])/len(items_ratings[item])'''


    return users_items, define_index(tokens['users']), define_index(tokens['items'])#, users_ratings, items_ratings


def define_prediction_features(prediction_data, modeling):

    tokens = utils.retrieve_unique_tokens(prediction_data)

    # a dictionary of the modeling tokens a empty lists
    tokens_info = dict(zip(tokens[modeling], [[]] * len(tokens[modeling])))

    if modeling == 'items':

        keys_index, values_index = 1, 0

    else:

        keys_index, values_index = 0, 1

    for row in prediction_data:

        tokens_info[row[keys_index]].append(row[values_index])

    return tokens_info


def mount_matrix_user_item(users_items):

    matrix = []

    for user, items in users_items.items():

        matrix.extend(list(map(lambda item: [user, item], items)))

    return matrix



def define_user_item_rating(historic_data):

    tokens = utils.retrieve_unique_tokens(historic_data)

    users_ratings = dict(zip(tokens['users'], [{}] * len(tokens['users'])))

    for row in historic_data:

        users_ratings[row[0]][row[1]] = float(row[2])

    return users_ratings


