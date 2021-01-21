"""

    Defining metrics that will be used to solve the problem

"""

import math


def select_similarity_metric(similarity_metric):
    """
        Selecting the similarity metric that is more suited to the problem

        return a function of the choosen similarity metric
    """

    if similarity_metric == 'cosine':

       return measure_cosine_similarity

    assert "Unknow Similarity Measure"


def measure_cosine_similarity(tokens_one, tokens_two):
    """
        Calculating the cosine similarity between two arrays

        return a float value between -1 and 1
    """

    # measuring the numerator step
    numerator = sum(list(map(lambda a, b: a * b, tokens_one, tokens_two)))

    # denominator from the first token
    denominator_one = math.sqrt(sum(list(map(lambda a: a**2, tokens_one))))

    # denominator from the second token
    denominator_two = math.sqrt(sum(list(map(lambda a: a**2, tokens_two))))

    return numerator/(denominator_one * denominator_two)

def measure_similarity(tokens_one, tokens_two, similarity_metric='cosine'):
    """
        Receives as input two arrays with tokens (items or users) ratings

        Example:

            If the modeling is by user (The rating that a user has given to itens): 

                User -> 3, 4, 1

            If the modeling is by item (The Rating that a item has receive from differnent users):
                
                item -> 9, 4, 3, 4, 0

        return a float value
    """

    similarity_method = select_similarity_metric(similarity_metric)

    return similarity_method(tokens_one, tokens_two)


def root_mean_squared(predicted, real):
    """
        Measure the root mean square between the real and predicted values

        Params:

            two arrays of lenght n, where the predicted value is given by a algorithm
            and the real value is retrieved from the historic dataset


        return a float value
    """


    if len(predicted) != len(real):

        assert "Predicted and Real arrays most have the same lenght !"


    return math.sqrt(sum(list(map(lambda y_pred, y_real: (y_pred - y_real)** 2))/len(predicted)))
