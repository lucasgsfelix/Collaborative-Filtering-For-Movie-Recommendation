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
