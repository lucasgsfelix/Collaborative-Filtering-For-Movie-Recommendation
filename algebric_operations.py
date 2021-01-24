"""

	Define all algebric operations

"""

import random

def transpose_matrix(matrix):
    """
        Given a matrix A, this method will return A Transposed

        A matrix will be a python list of list

    """

    transposed_matrix = []

    # matrix[aqui varia][aqui é fixo]

    for column in range(0, len(matrix[0])):

        new_row = []

        for row in range(0, len(matrix)):

            new_row.append(matrix[row][column])

        transposed_matrix.append(new_row)

    return transposed_matrix

def matrix_multiplication(matrix_a, matrix_b):
    """

        Given two matrixes, this method returns a third matrix that
        is the multiplication of the two entry matrixes

    
        Row X Column

    """

    result_matrix = []

    for row in matrix_a:

        # I have a list of values that will be multiplied by the column of another matrix

        new_row = []

        for k in range(0, len(matrix_a)): # vary the rows for matrix b

            total = 0

            for j in range(0, len(matrix_b)): # Vary the columns for the matrix_b

                total += matrix_b[j][k] * row[j]

            new_row.append(total)

        result_matrix.append(new_row)

    return result_matrix

def define_matrix_determinant(matrix, lambda_value):


    main_diagonal = 1

    for i in range(0, len(matrix)):

        main_diagonal = main_diagonal * (matrix[i][i] - lambda_value)

    return main_diagonal

def generate_random_matrix(row, columns, min_value=0, max_value=1):


    return [[random.uniform(min_value, max_value) for column in range(columns)] for row_index in range(0, row)]

def sum_two_arrays(array_one, array_two):

    if len(array_one) != len(array_two):

        assert "The arrays most have the same size !"

    return list(map(lambda a, b: a + b, array_one, array_two))

