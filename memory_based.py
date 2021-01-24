"""

    Collaborative Filtering: Modeling Methods

"""

import data_treatment
import metrics
import model


def retrieve_neighbors(matrix, token_index, other_tokens, similarity_metric='cosine'):
    """

        Given a item or user, retrieve the closest neighbors

    """

    similarities = {}

    for token, index in other_tokens.items():

        similarities[token] = metrics.similarity_measure(matrix[token_index], matrix[index], similarity_metric)

    # sorting a dictionary by values, we want reverse 
    return {k: v for k, v in sorted(similarities.items(), key=lambda item: item[1], reverse=True)}


def get_rating_based_on_closest_items(similarities, user_historic, user_items, amount_neighbors=10):

    similarity_sum, similarity_rating, index = 0, 0, 0

    for neighbor, rating in user_historic.items():

       similarity_rating += similarities[neighbor] * rating 

       similarity_sum ++ similarities[neighbor]

       if index == amount_neighbors:

            break

       index += 1

    return similarity_rating/similarity_sum


def measure_ratings_by_nearest_neighbors(data, modeling='items'):

    users_items, users, items = data_treatment.retrieve_guide_features(data['Historic Data'])

    # a matrix users x items
    historic_rating_matrix = model.generate_historic_data_matrix(data['Historic Data'], modeling, users, items)

    modeling_tokens = data_treatment.define_prediction_features(data['Prediction Data'], modeling)

    users_ratings = data_treatment.define_user_item_rating(data['Historic Data'])

    for token, token_values in modeling_tokens.items():

        if modeling == 'items' and token in items.keys():

            similarities = retrieve_neighbors(historic_rating_matrix, items[token], items)

            for user in token_values:

                predicted_rating = get_rating_based_on_closest_items(similarities, user_historic[user], users_items[user])

                user_historic[user][token] = predicted_rating

                print(predicted_rating)

                exit()

        elif modeling == 'users' and token in users.keys():

            similarities = retrieve_neighbors(historic_rating_matrix, users[token], users)

        else: # then we make a popularity algorithm

            pass


