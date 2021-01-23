"""

    Collaborative Filtering: Modeling Methods

"""
import os
import metrics
import utils

import time


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

    # the similarity matrix will be computed only for the items in the test set

    columns = {token_two: 0 for token_two in tokens[modeling].keys()}

    similarity_matrix = {token_one: {} for token_one in test_tokens[modeling].keys()}


    if verify_pre_computed_similarity_matrix(similarity_metric):

        similarity_matrix = utils.read_table("Utils/similarity_" + similarity_metric + "_matrix.txt", sep=';')

        # converting the similarity to float
        for index, row in enumerate(similarity_matrix):

            similarity_matrix[index] = list(map(lambda x: float(x), row))

    else:

        for token_one in test_tokens[modeling].keys():

            start = time.time()

            index_one = test_tokens[modeling][token_one]

            print("Current Token: ", token_one)

            for token_two in tokens[modeling].keys():

                index_two = tokens[modeling][token_two]

                if token_one == token_two:

                    similarity_matrix[token_one][token_two] = 1
                else:

                    similarity_matrix[token_one][token_two] = metrics.measure_similarity(matrix[index_one], matrix[index_two], similarity_metric)

            print(time.time() - start)

            similarity_matrix[token_one] = {k: v for k, v in sorted(similarity_matrix[token_one].items(), key=lambda item: item[1], reverse=True)}[0:100]        

        utils.write_dictionary_matrix(similarity_matrix, "Utils/similarity_" + similarity_metric + "_matrix.txt", sep=';')

    return similarity_matrix

def measure_nearest_neighbors(data_matrix, k_neighbors):

    pass

