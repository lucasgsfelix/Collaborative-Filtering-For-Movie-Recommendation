"""

	Defining the operations for a model two work

"""


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
