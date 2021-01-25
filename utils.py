"""

    Utility methods for movie recommendation

"""

def read_table(file_input, sep=':', replace_char=None):
    """
        Read table with data

        return a list of lists
    """

    with open(file_input, 'r') as read_input:

        data = read_input.read()

        if replace_char:

            data = data.replace(replace_char, sep)

        data = data.split('\n')

        # removing the header
        data.pop(0)

    data = list(map(lambda row: row.split(sep), data))

    return list(filter(lambda row: row[0] != '', data))


def write_dictionary(dictionary, output_name, sep=';'):

    with open(output_name, 'w') as output_file:

        for key, value in dictionary.items():

            output_file.write(sep.join([str(key), str(value)]) + '\n')


def write_dictionary_matrix(dictionary_matrix, output_name, sep=';'):

    with open(output_name, 'w') as output_file:

        for key, row in dictionary.items():

            output_file.write(key + sep)

            for key_id, value in row.items():

                output_file.write(str(key_id) + sep + str(value))

        output_file.write('\n')


def write_table(matrix, output_name):

    with open(output_name, 'w') as output_file:

        for row in matrix:

            output_file.write(row[0] + ':' + row[1] + ',' + row[2] + '\n')

def retrieve_unique_tokens(data):
    """
        Retrieve the unique tokens to model the matrix

        return a dictionary of token and its unique items

    """

    tokens = {"users": list(set(list(map(lambda row: row[0], data)))),
              "items": list(set(list(map(lambda row: row[1], data))))}

    return tokens


def write_array(data, output_name):

    with open(output_name, 'w') as output_file:

        data = list(map(lambda x: str(x), data))

        output_file.write('\n'.join(data))


def measure_row_mean(matrix):
    """
        Measure each row mean in a matrix
    """

    rows_mean = {}

    for index, row in enumerate(matrix):

        non_zero_row = list(filter(lambda x: x != 0, row))

        rows_mean[index] = sum(non_zero_row)/len(non_zero_row)

    return rows_mean


def measure_column_mean(matrix):
    """
        Measure each column mean in a matrix
    """

    columns_mean = {}

    for column in range(len(matrix[0])):

        count_valid, total_sum = 0, 0

        for row in range(len(matrix)):

            total_sum += matrix[row][column]

            if matrix[row][column] != 0: # removing rows that are equal to 0

                count_valid += 1

        columns_mean[column] = total_sum/count_valid


def subtraction_matrix_row_mean(matrix, rows_mean):

    for row_index, row in enumerate(matrix):

        for column_index, column in enumerate(row):

            matrix[row_index][column_index] -= rows_mean[row_index]

    return matrix
