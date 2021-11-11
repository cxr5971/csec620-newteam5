import numpy
import pandas
import torch
import sys
from vector_builder import prepare_data, prepare_data_ova

DATA_DIRECTORY = "dataset/feature_vectors.zip"
LABEL_FILE = "dataset/sha256_family.csv"
TRAINING_RATIO = .7
TESTING_RATIO = 1 - TRAINING_RATIO
BENIGN_RATIO = 1


def train_SVM_OVA(features: set, training_data: dict, labels: dict,
              epochs: int, learning_rate: float, C: int, target_label: str):
    """
    This function trains the SVM for One vs. All predictions. It loops for the specified number of epochs and in each epoch it loops over every
    point in the training data, predicts if the sample is associated with the target malware family, calculates the hinge loss, and then
    performs gradient descent to tune the slope (w) and y intercept (b) of our linear SVM. This will be used in a loop to get training models for each malware family
    :param features: set: All the features used for training
    :param training_data: dict: A dictionary containing all the training samples and their feature vectors
    :param labels: dict: a dictionary of malicious samples and their labels
    :param epochs: int: The number of epochs to train the SVM for
    :param learning_rate: int: The learning rate to use for our gradient descent
    :param C: int: The weight of our classification error. Would you like a hard or soft margin?
    :param target_label: str: This is the malware family we are trying to classify
    :return: The trained SVM parameters w and b
    """

    w = torch.rand(len(features), requires_grad=True, dtype=torch.float)
    b = torch.rand(1, requires_grad=True, dtype=torch.float)

    for epoch in range(epochs):
        print(epoch)
        for sample, vector in training_data.items():
            prediction = torch.dot(w, torch.as_tensor(numpy.array(vector), dtype=torch.float)) - b
            true_value = -1 if labels[sample] != target_label else 1
            max_value = max(0, 1 - (true_value * prediction)) 
            hinge_loss = max_value + (C * (torch.linalg.norm(w)/2))

            if max_value != 0:
                hinge_loss.backward()
                w.data -= learning_rate * w.grad.data
                b.data -= learning_rate * b.grad.data
                w.grad.data.zero_()
                b.grad.data.zero_()
    return w, b



def train_SVM(features: set, training_data: dict, labels: dict,
              epochs: int, learning_rate: float, C: int):
    """
    This function trains the SVM. It loops for the specified number of epochs and in each epoch it loops over every
    point in the training data, predicts whether it is benign or malicious, calculates the hinge loss, and then
    performs gradient descent to tune the slope (w) and y intercept (b) of our linear SVM
    :param features: set: All the features used for training
    :param training_data: dict: A dictionary containing all the training samples and their feature vectors
    :param labels: dict: a dictionary of malicious samples and their labels
    :param epochs: int: The number of epochs to train the SVM for
    :param learning_rate: int: The learning rate to use for our gradient descent
    :param C: int: The weight of our classification error. Would you like a hard or soft margin?
    :return: The trained SVM parameters w and b
    """

    # Initialize our w and b vectors using pytorch. Random because why not
    w = torch.rand(len(features), requires_grad=True, dtype=torch.float)
    b = torch.rand(1, requires_grad=True, dtype=torch.float)

    # let the time sink begin
    for epoch in range(epochs):
        for sample, vector in training_data.items():
            prediction = torch.dot(w, torch.as_tensor(numpy.array(vector), dtype=torch.float)) - b
            true_value = -1 if sample in labels.keys() else 1  # if it is in labels it must be malicious
            hinge_loss = max(0, 1 - (true_value * prediction)) + (C * (torch.linalg.norm(w)/2))

            if hinge_loss != 0:  # if its zero everything breaks
                hinge_loss.backward()  # why couldn't calc 2 derivatives be this simple
                w.data -= learning_rate * w.grad.data  # One small step for SVM. No I mean its really tiny
                b.data -= learning_rate * b.grad.data
                w.grad.data.zero_()  # reset the gradients to remove residual effects
                b.grad.data.zero_()

        # Just a simple sanity check
        if (epoch+1) % (epochs/10) == 0:
            print(f"Training: {((epoch+1)/epochs) * 100}% complete")
    return w, b


# loop all samples
# loop all the models
# predict
# get true value

def confusion_matrix(epoch, features, training_data, testing_data, labels, malware_families):
    """
    Loops over the top 20 malware families, trains and predicts on the One vs. All classifier
    to measure our accuracy in correctly classifying samples among many classes.
    :param epoch: int: Number of iterations to use in training
    :param features: set: All the features in the training data
    :param training_data: dict: A dictionary mapping the training data to their feature vectors
    :param testing_data: dict: A dictionary mapping the testing data to their feature vectors
    :param labels: dict: A dictionary mapping each sample to its true classification label
    :param malware_families: dict: A dictionary mapping each malware family to a list of samples
    :return: A dataframe containing the confusion matrix
    """
    top_20 = list(malware_families.items())
    top_20.sort(key=lambda x: len(x[1]), reverse=True)
    #print(top_20)
    models = {}
    count = 0
    for family in top_20:
        if count == 20:
            break
        print("Malware Family: " + family[0])
        models[family[0]] = train_SVM_OVA(features, training_data, labels, epoch, .001, 1, family[0])
        count += 1

    sample_predictions = predict_ova(testing_data, models)
   


    matrix = []
    family_label_list = []

    for family in top_20[0:20]:
        family_label_list.append(family[0])
    
    matrix.append([" "]+family_label_list)
    for family in top_20[0:20]:
        count = 0
        #Malware predictions keeps track of everytime we predict that a sample is some malware
        #key: label, value: number of samples predicted
        malware_predictions = dict.fromkeys(family_label_list,0)
        for sample in family[1]:
            if sample in sample_predictions.keys():
                count += 1
                predicted_value = sample_predictions[sample]
                malware_predictions[predicted_value] += 1
        for value in malware_predictions.keys():
            malware_predictions[value] = malware_predictions[value]/count
        matrix.append([family[0]] + list(malware_predictions.values()))
    df = pandas.DataFrame(matrix)
    return(df)


def predict_ova(test_data, models):
    """
    Loops over all the test samples and predicts what class they are. Counts the number of correct and
    incorrect predictions and returns them
    :param test_data: dict: A dictionary mapping the test data to their feature vectors
    :param models: dict: contains the training model generated for each malware family 
    :return: Dictionary containing samples as keys and the proposed prediction label as values
    """
    sample_predictions = {}
    for sample, vector in test_data.items():
        class_confidence = -9000
        family_value = None
        for family, model in models.items():
            prediction = torch.dot(model[0], torch.as_tensor(numpy.array(vector), dtype=torch.float)) - model[1]

            if prediction > class_confidence:
                class_confidence = prediction
                family_value = family
        sample_predictions[sample] = family_value

    return sample_predictions


def predict(test_data: dict, w: torch.Tensor, b: torch.Tensor, labels: dict):
    """
    Loops over all the test samples and predicts whether they are benign or malicious. Counts the number of correct and
    incorrect predictions and returns them
    :param test_data: dict: A dictionary mapping the test data to their feature vectors
    :param w: torch.Tensor: The trained SVM parameter w
    :param b: torch.Tensor: The trained SVM parameter b
    :param labels: dict: A dictionary mapping the malicious samples to their malware family
    :return: The count of correct and incorrect predictions
    """
    correct = 0
    wrong = 0

    # Essentially the core of the SVM just without the looping and loss
    for sample, vector in test_data.items():
        prediction = torch.dot(w, torch.as_tensor(numpy.array(vector), dtype=torch.float)) - b
        true_value = -1 if sample in labels.keys() else 1

        # We only care if its in the correct class so use numpy.sign to convert any value to either 1 or -1
        if true_value == numpy.sign(prediction.detach().numpy()[0]):
            correct += 1
        else:
            wrong += 1

    return correct, wrong


def hyper_tuning_svm(features: set, training_data: dict, testing_data: dict, labels: dict,
                     epochs: int, learning_rate: float, C: int):
    """
    A modified form of the SVM to collect data for the hyper parameter tuning. Essentially it predicts and checks
    the accuracy each epoch to gather large amounts of data without needing to repeat as many times
    :param features: set: All the features in the training data
    :param training_data: dict: A dictionary mapping the training data to their feature vectors
    :param testing_data: dict: A dictionary mapping the testing data to their feature vectors
    :param labels: dict: A dictionary that maps malicious samples to their malware family. Really only use the keys
    :param epochs: int: The number of iterations to train the SVM
    :param learning_rate: int: The learning rate of gradient descent
    :param C: int: The weight of the classification error for soft margins
    :return: The trained model parameters w and b
    """

    # This seems very familar
    w = torch.rand(len(features), requires_grad=True, dtype=torch.float)
    b = torch.rand(1, requires_grad=True, dtype=torch.float)

    for epoch in range(1, epochs+1):
        for sample, vector in training_data.items():
            prediction = torch.dot(w, torch.as_tensor(numpy.array(vector), dtype=torch.float)) - b
            true_value = -1 if sample in labels.keys() else 1
            hinge_loss = max(0, 1 - (true_value * prediction)) + (C * (torch.linalg.norm(w)/2))

            if hinge_loss != 0:
                hinge_loss.backward()
                w.data -= learning_rate * w.grad.data
                b.data -= learning_rate * b.grad.data
                w.grad.data.zero_()
                b.grad.data.zero_()

        # I heard you like data.
        # There used to be an if to control when to print based on a step but we got lazy
        print(f"running with {epoch} epochs")
        correct, wrong = predict(testing_data, w, b, labels)

        print(f"w: {w}\nb: {b}")
        print(f"Correct: {correct}")
        print(f"Wrong: {wrong}")
        print(f"Accuracy: {correct/(correct + wrong)}")

    return w, b


def find_best_and_worst(features: set, w: torch.Tensor):
    """
    This function prints the top and bottom 10 features according to the model
    :param features: set: The set of all features used in training the model
    :param w: torch.Tensor: The w vector of the SVM
    :return: None
    """
    feature_weights = dict(zip(features, w))  # pair the features and their w value so they don't get lost in sorting
    feature_weights = list(feature_weights.items())
    feature_weights.sort(key=lambda tup: tup[1])  # fancy lambda sorting because Im too lazy to write another function
    print(f"Top 10 features:\n\t{feature_weights[-10:]}")  # I was also too lazy to reverse the list so weird slicing
    print(f"Bottom 10 features:\n\t{feature_weights[:10]}")


def linear_svm(data_directory: str, label_file: str, training_ratio: int, benign_ratio: int):
    """
    The linear SVM. This function loads the data set, processes it, trains the SVM, tests the model, and reports the
    results. Now with print statements
    :param data_directory: str: The directory that contains all the samples
    :param label_file: str: The name/path of the file with all the malicious data's labels
    :param training_ratio: int: The percent of malicious data to use in the training data
    :param benign_ratio: int: The ratio of benign to malicious samples that should make up the training/testing data
    :return: None
    """
    print("Starting to process data")
    features, training_data, testing_data, labels, malware_families = prepare_data(data_directory, label_file,
                                                                                   training_ratio, benign_ratio)
    print("Finished loading data\nTraining SVM")
    w, b = train_SVM(features, training_data, labels, 100, .001, .04)
    print("Training complete\nBeginning predictions")
    correct, wrong = predict(testing_data, w, b, labels)

    print("Finished predictions\nResults:")
    print(f"Correct: {correct}")
    print(f"Wrong: {wrong}")
    print(f"Accuracy: {correct/(correct + wrong)}")
    find_best_and_worst(features, w)


def main():
    #If you specify -ova on the command line, SVM will perform using the One vs. All classifier (malware family classification
    #Otherwise, SVM runs classifying malware vs. benign
    if len(sys.argv) > 1 and "-ova" in sys.argv:
        features, training_data, testing_data, labels, malware_families = prepare_data_ova(DATA_DIRECTORY, LABEL_FILE,
                                                                                           TRAINING_RATIO, BENIGN_RATIO)

        epoch_tests = [20, 40, 50, 70, 100]
        for epoch in epoch_tests:
            df = confusion_matrix(epoch, features, training_data, testing_data, labels, malware_families)
            #Saves the confusion matrix to a file, we put this file into excel to allow for easier manipulation
            file_name = "out_" + str(epoch) + ".file"
            df.to_csv(file_name, index=False)
    else:
        linear_svm(DATA_DIRECTORY, LABEL_FILE, TRAINING_RATIO, BENIGN_RATIO)


if __name__ == '__main__':
    main()


