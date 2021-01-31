"""

    Test Cases

"""

import main_assignment


combinations = [(2, 10), (5, 10), (10, 10), (20, 10), (2, 20), (5, 20), (20, 20), (2, 30), (5, 30), (20, 30)]

#for combination in combinations:

#    latent_factos, epochs = combination

#    print("Testing : ", combination)

 #   main_assignment.recommender_system(latent_factos, epochs, True)

for combination in combinations:

    latent_factos, epochs = combination

    for iteration in range(0, 5):

        print("Evaluating Time: ", latent_factos, epochs, iteration)

        file_name = "prediction_" + str(latent_factos) + "_" + str(epochs) + "_" + str(iteration) + ".txt"

        main_assignment.recommender_system(latent_factos, epochs, False, file_name)
