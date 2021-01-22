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


def write_table(matrix, output_name, sep=';'):

    with open(output_name, 'w') as output_file:

        for row in matrix:

            output_file.write(sep.join(list(map(lambda x: str(x), row))) + '\n')
