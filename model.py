"""

    Collaborative Filtering: Modeling Methods

"""
import metrics


def generate_historic_data_matrix(historic_data, modeling='item'):
    """
        Modeling the matrix of historical data

        Modeling:

            Define if the matrix generate will be a item x item or user x user modeling


    """


    tokens = retrieve_unique_tokens(historic_data)

    # making a matrix of zeros
    if modeling == 'item':

        matrix = [[0] * len(tokens['users'])] * len(tokens['items'])

    else:

        matrix = [[0] * len(tokens['items'])] * len(tokens['users'])

    for row in historic_data:

        user = tokens['users'].index(row[0])
        item = tokens['items'].index(row[1])

        # rating given by the user
        rating = int(row[2])

        if modeling == 'item':

            matrix[item][user] = rating

        else:

            matrix[user][item] = rating

    return matrix, tokens

def model_similarity_matrix(historic_data, similarity_metric='cosine', modeling='item'):
    """

        Modeling the similarity matrix between all tokens (users, itens)

        return similarity_matrix

    """

    matrix, tokens = generate_historic_data_matrix(historic_data, modeling)

    amount_rows = len(tokens['items'])

    # the amount of rows and columns will be the same
    similarity_matrix = [[0] * amount_rows] * amount_rows

    verified_tuples = []

    for index_one, row_one in enumerate(matrix):

        for index_two, row_two in enumerate(matrix):

            if index_one == index_two:

                similarity_matrix[index_one][index_two] = 1

            elif tuple(index_two, index_one) in verified_tuples:

                similarity_matrix[index_one][index_two] = similarity_matrix[index_two][index_one]

            else:

                similarity_matrix[index_one][index_two] = metrics.measure_similarity(row_one, row_two, similarity_metric)

                verified_tuples.append(tuple(index_one, index_two))

    return similarity_matrix

def k_nearest_neighbors(data_matrix, k_neighbors):

    pass


def retrieve_unique_tokens(data):
    """
        Retrieve the unique tokens to model the matrix

        return a dictionary of token and its unique items

    """

    return {"users": list(set(list(map(lambda row: row[0], data)))),
            "items": list(set(list(map(lambda row: row[1], data))))}
