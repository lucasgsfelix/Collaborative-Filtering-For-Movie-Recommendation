"""

    Data Treatment and Utility Functions for data Identification

"""

import utils

def define_index(tokens):

    return {token: index for index, token in enumerate(tokens)}


def retrieve_guide_features(historic_data):

    tokens = utils.retrieve_unique_tokens(historic_data)

    users_items = {user: [] for user in tokens['users']}

    for row in historic_data:

        users_items[row[0]].append(row[1])


    return users_items, define_index(tokens['users']), define_index(tokens['items'])



def mount_matrix_user_item(users_items):

    matrix = []

    for user, items in users_items.items():

        matrix.extend(list(map(lambda item: [user, item], items)))

    return matrix
