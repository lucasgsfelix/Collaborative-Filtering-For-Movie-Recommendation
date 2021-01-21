"""

	First programming assignement of Recommender Systems class of 2020/2 - UFMG

    Our aim is to make a collaborative filtering for move recommendation

    Made by: Lucas Félix
    E-Mail: lucasgsfelix@gmail.com

"""

import sys


def read_table(file_input, sep=':', replace_char=None):

    with open(file_input, 'r') as read_input:

        data = read_input.read()

        if replace_char:

            data = data.replace(replace_char, sep)

        data = data.split('\n')

    return list(map(lambda row: row.split(sep), data))


def retrieve_unique_tokens(data):


    return {"users": list(set(list(map(lambda row: row[0], data)))),
            "itens": list(set(lis(map(lambda row: row[1], data))))}


def model_matrix():

    pass


def k_nearest_neighbors(data_matrix, k_neighbors):

    pass


if __name__ == '__main__':


    input_arguments = {"Historic Data": read_table(sys.argv[1]),
                       "Prediction Data": read_table(sys.argv[2], ':', ',')}




