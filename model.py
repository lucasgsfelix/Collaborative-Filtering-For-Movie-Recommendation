"""

    Collaborative Filtering: Modeling Methods

"""
import os
import metrics
import utils


def generate_historic_data_matrix(historic_data, modeling='item'):
    """
        Modeling the matrix of historical data

        Modeling:

            Define if the matrix generate will be a item x item or user x user modeling


    """


    tokens = retrieve_unique_tokens(historic_data)

    # making a matrix of zeros
    if modeling == 'items':

        matrix = [[0] * len(tokens['users']) for row in range(0, len(tokens['items']))]

    else:

        matrix = [[0] * len(tokens['items']) for row in range(0, len(tokens['users']))]

        #matrix = [[0] * len(tokens['items'])] * len(tokens['users'])

    for row in historic_data:

        user = tokens['users'][row[0]]
        item = tokens['items'][row[1]]

        # rating given by the user
        rating = int(row[2])

        if modeling == 'items':

            matrix[item][user] = rating

        else:

            matrix[user][item] = rating

    return matrix, tokens


def verify_pre_computed_similarity_matrix(similarity_metric):

    return "similarity_" + similarity_metric + "_matrix.txt" in os.listdir('Utils')

def model_similarity_matrix(data, similarity_metric='cosine', modeling='items'):
    """

        Modeling the similarity matrix between all tokens (users, itens)

        return similarity_matrix

    """
    matrix, tokens = generate_historic_data_matrix(data['Historic Data'], modeling)

    test_tokens = retrieve_unique_tokens(data['Prediction Data'])

    # removing users with no historic
    not_historic_users = list(filter(lambda x: x not in tokens['users'].keys(), test_tokens['users'].keys()))
    test_tokens['users'] = {key: value for key, value in test_tokens['users'].items() if key in tokens['users'].keys()}

    amount_rows = len(test_tokens[modeling])

    # the similarity matrix will be computed only for the items in the test set
    similarity_matrix = [[None] * len(tokens[modeling])] * amount_rows

    if verify_pre_computed_similarity_matrix(similarity_metric):

        similarity_matrix = utils.read_table("Utils/similarity_" + similarity_metric + "_matrix.txt", sep=';')

        # converting the similarity to float
        for index, row in enumerate(similarity_matrix):

            similarity_matrix[index] = list(map(lambda x: float(x), row))

    else:

        for index_one, token_one in enumerate(test_tokens[modeling].keys()):

            for index_two, token_two in enumerate(tokens[modeling].keys()):

                index_one = test_tokens[modeling][token_one]
                index_two = tokens[modeling][token_two]

                new_index_one, new_index_two = None, None

                if token_two in test_tokens[modeling].keys() and token_one in tokens[modeling].keys():

                    # getting the row
                    new_index_one = test_tokens[modeling][token_two]

                    # getting the column
                    new_index_two = tokens[modeling][token_one]


                if token_one == token_two:

                    similarity_matrix[index_one][index_two] = 1

                elif (similarity_matrix[index_one][index_two] is not None or
                     (new_index_one is None and new_index_two is not None and similarity_matrix[new_index_one][new_index_two] is not None)):

                    continue

                else:

                    similarity_matrix[index_one][index_two] = metrics.measure_similarity(matrix[index_one], matrix[index_two], similarity_metric)

                    if new_index_one is not None or new_index_two is not None:

                        similarity_matrix[new_index_one][new_index_two] = similarity_matrix[index_one][index_two]



        utils.write_table(similarity_matrix, "Utils/similarity_" + similarity_metric + "_matrix.txt", sep=';')

    return similarity_matrix

def measure_nearest_neighbors(data_matrix, k_neighbors):

    pass


def retrieve_unique_tokens(data):
    """
        Retrieve the unique tokens to model the matrix

        return a dictionary of token and its unique items

    """

    tokens = {"users": list(set(list(map(lambda row: row[0], data)))),
              "items": list(set(list(map(lambda row: row[1], data))))}


    tokens['users'] = {user_id: index for index, user_id in enumerate(tokens['users'])}
    tokens['items'] = {item_id: index for index, item_id in enumerate(tokens['items'])}

    return tokens