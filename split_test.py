"""
	
	Test Data

"""

import time
import metrics
import model_based_nmf
import statistics

def split_test(input_arguments, latent_size, epochs):


    paramters_file = open("Outputs/relatorio.csv", "a+")

    k_folds = 5

    for index, row in enumerate(input_arguments['Historic Data']):

        input_arguments['Historic Data'][index][3] = float(row[3])

    # sorting by the timestamp
    input_arguments['Historic Data'].sort(key=lambda row: row[3])

    data_per_fold = int(len(input_arguments['Historic Data'])/k_folds)
    total_data = len(input_arguments['Historic Data'])


    total_rmse = []

    for fold in range(k_folds):

        data = {}

        print("Fold: ", fold, latent_size, epochs)

        start = time.time()

        if fold == 0:

            data['Prediction Data'] = input_arguments['Historic Data'][0: data_per_fold]
            data['Historic Data'] = input_arguments['Historic Data'][data_per_fold+1: total_data]

        elif fold > 0 and fold < 10:

            data['Prediction Data'] = input_arguments['Historic Data'][fold * data_per_fold: fold * data_per_fold * 2]

            first_part = input_arguments['Historic Data'][0: (fold * data_per_fold) - 1]
            last_part = input_arguments['Historic Data'][(fold * data_per_fold * 2) + 1: total_data]
            data['Historic Data'] = first_part + last_part

        elif fold == 10:

            data['Prediction Data'] = input_arguments['Historic Data'][total_data - data_per_fold: total_data]
            data['Historic Data'] = input_arguments['Historic Data'][0: total_data - data_per_fold - 1]

        predictions, epochs_rmse = model_based_nmf.non_negative_matrix_factorization(data, latent_size, epochs, None, True)

        real_values = list(map(lambda row: float(row[2]), input_arguments['Historic Data']))
        rmse = metrics.root_mean_squared(predictions, real_values)

        total_rmse.append(rmse)


    paramters_file.write('\t'.join([str(latent_size), str(epochs), str(k_folds), str(statistics.mean(total_rmse)),
                                   (str(statistics.stdev(total_rmse))), str(time.time() - start)]) + '\n')

    paramters_file.close()
