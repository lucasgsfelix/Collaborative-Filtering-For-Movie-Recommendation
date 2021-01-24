"""

    Collaborative Filtering: Modeling Methods

"""
import os
import metrics
import utils

import time


def matrix_factorization(matrix):
    """
        Given a matrix that store ratings the idea is to estimate unknown ratings by factorizing the
        rating matrix into two smaller matrices representing user and item characteristics

    """

    users_items, users, items = data_treatment.retrieve_guide_features(data['Historic Data'])

    matrix_users_items = data_treatment.mount_matrix_user_item(users_items)

    # a matrix users x items
    historic_rating_matrix = model.generate_historic_data_matrix(data['Historic Data'], 'users', users, items)


