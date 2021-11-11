from sklearn_vector_builder import prepare_data
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn import svm

# File locations for the data set
DATA_DIRECTORY = "dataset/feature_vectors.zip"
LABEL_FILE = "dataset/sha256_family.csv"
# Ratios for split between training and testing
TRAINING_RATIO = .7
TESTING_RATIO = 1 - TRAINING_RATIO
# Reduction in set size
BENIGN_RATIO = .8
MALWARE_RATIO = .8


def makeData(training_data: dict, testing_data: dict, labels: dict):
    """
    A simple function that converts the dictionary data into usable data for SVM
    :param training_data: Dict of training data
    :param testing_data: Dict of testing data
    :param labels: Dict of malware lables
    :return: xtrain, xtest, ytrain, ytest The training and testing sets
    """
    ytrain = []
    xtrain = list(training_data.values())
    for sample, vector in training_data.items():
        if sample in labels.keys():
            ytrain.append(-1)
        else:
            ytrain.append(1)
    ytest = []
    xtest = list(testing_data.values())
    for sample, vector in testing_data.items():
        if sample in labels.keys():
            ytest.append(-1)
        else:
            ytest.append(1)
    return xtrain, ytrain, xtest, ytest


# sklearn implementation of LinearSVM
def linear_SVM(training_data: dict, labels: dict, testing_data: dict):
    lsvm = LinearSVC(verbose=0)
    xtrain, ytrain, xtest, ytest = makeData(training_data, testing_data, labels)
    lsvm.fit(xtrain, ytrain)
    ypred = lsvm.predict(xtest)
    cm = confusion_matrix(ytest, ypred)
    print(cm)
    cr = classification_report(ytest, ypred)
    print(cr)


# sklearn implementation of SVM with rbf kernel
def rbf_SVM(training_data: dict, testing_data: dict, labels: dict):
    nsvm = svm.SVC(kernel='rbf')
    xtrain, ytrain, xtest, ytest = makeData(training_data, testing_data, labels)
    nsvm.fit(xtrain, ytrain)
    ypred = nsvm.predict(xtest)
    cm = confusion_matrix(ytest, ypred)
    print(cm)
    cr = classification_report(ytest, ypred)
    print(cr)


def main():
    training_data, testing_data, labels = prepare_data(DATA_DIRECTORY, LABEL_FILE, TRAINING_RATIO, BENIGN_RATIO,
                                                       MALWARE_RATIO)

    linear_SVM(training_data, labels, testing_data)
    rbf_SVM(training_data, testing_data, labels)


if __name__ == '__main__':
    main()
