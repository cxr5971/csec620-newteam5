import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import model_selection, linear_model, metrics, svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.inspection import permutation_importance


def prepare_data(filename, benign_ratio, drop_cols, malicious_label, test_size=.33, seed=123456):
    """
    Loads in preprocessed data from file and then undersamples the data. Once undersampled, the labels are grabbed,
    final columns are removed, and the data is split into training and testing sets
    :param filename: str: The file of preprocessed data
    :param benign_ratio: int: The ratio of benign to malicious samples to use
    :param drop_cols: list[str]: A list of columns to drop from the data. Labels_<lambda> is always dropped so don't
        include it in the list
    :param malicious_label: int: The label of the malicious class
    :param test_size: float: The amount of data to use for the training set
    :param seed: int: A seed for how to randomly pick samples
    :return: All the training and testing sets
    """
    print(f"Loading data from {filename}")
    all_data = pd.read_hdf(filename, key='data')
    all_data.reset_index(drop=True, inplace=True)
    print(f"Undersampling the data with a {benign_ratio}:1 split")
    undersampled_data = undersample_data(all_data, benign_ratio, seed)

    # Drop any unwanted columns from the data
    for column in drop_cols:
        undersampled_data.drop(column, axis=1, inplace=True)

    # NO MATTER WHAT REMOVE THE LABELS
    labels = undersampled_data['Label_<lambda>']
    undersampled_data.drop('Label_<lambda>', axis=1, inplace=True)

    # Convert traffic type to malious or benign (true or false)
    cleaned_labels = labels == malicious_label
    print("Creating training and test sets")
    x_train, x_test, y_train, y_test = model_selection.train_test_split(undersampled_data, cleaned_labels,
                                                                        test_size=test_size, random_state=seed)
    return x_train, x_test, y_train, y_test


def undersample_data(data, benign_ratio, seed):
    """
    Undersamples the benign traffic
    :param data: pandas:Dataframe: All the data
    :param benign_ratio: int: The ratio of benign to normal traffic
    :param seed: int: A random seed
    :return: the undersampled data
    """
    malicious = data.loc[data['Label_<lambda>'] == 6]
    benign = data[data['Label_<lambda>'] != 6]
    mal_count = malicious.shape[0]
    bengin_count = int(mal_count * benign_ratio)
    undersampled_data = pd.concat([malicious, benign.sample(n=bengin_count, random_state=seed)])
    return undersampled_data


def report_metrics(y_test, y_pred):
    """
    Generates all the metrics for the provided data
    :param y_test: The true labels of the test data
    :param y_pred: The predicted labels of the test data
    :return: None
    """
    print("\tAccuracy:", metrics.accuracy_score(y_test, y_pred))
    print("\tPrecision:", metrics.precision_score(y_test, y_pred))
    print("\tRecall:", metrics.recall_score(y_test, y_pred))
    print("\tF1-Score:", metrics.f1_score(y_test, y_pred))
    tn, fp, fn, tp = metrics.confusion_matrix(y_test, y_pred).ravel()
    print("\tTrue Negative: " + str(tn))
    print("\tFalse Positive: " + str(fp))
    print("\tFalse Negative: " + str(fn))
    print("\tTrue Positive: " + str(tp) + "\n")


def tune_forrest(x_train, y_train, x_test, y_test, sampling_ratio, mini, maxi, step, filename):
    """
    Creates and runs through multiple iterations of the random forest model. Saves all output to a file.
    :param x_train: The training data
    :param y_train: The training labels
    :param x_test: The test data
    :param y_test: The test labels
    :param mini: Minimum of loop iterator, configure this for what parameter you want to test
    :param maxi: Maximum of loop iterator, configure for what parameter you want to test
    :param: step: Step value for testing loop
    :param: filename: The output file to save results
    :return: None
    """
    with open(filename, 'w+') as f_name:
        f_name.write('Accuracy,Precision,Recall,F1-Score,AUC,True_Negative,False_Positive,'
                     'False_Negative,True_Positive\n')
        for i in range(mini, maxi, step):
            f_name.write(str(i))
            print(f"Random Forest Classifier: iteration {i}")
            clf = RandomForestClassifier(n_estimators=i)
            clf.fit(x_train, y_train)
            y_pred = clf.predict(x_test)

            data = str(metrics.accuracy_score(y_test, y_pred)) + ','
            data += str(metrics.precision_score(y_test, y_pred)) + ','
            data += str(metrics.recall_score(y_test, y_pred)) + ','
            data += str(metrics.f1_score(y_test, y_pred)) + ','
            tn, fp, fn, tp = metrics.confusion_matrix(y_test, y_pred).ravel()
            fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred)
            data += str(metrics.auc(fpr, tpr)) + ','

            data += str(tn) + ','
            data += str(fp) + ','
            data += str(fn) + ','
            data += str(tp)
            f_name.write(data+'\n')


def random_forest(x_train, y_train, x_test, y_test, trees=100, importance=False):
    """
    Creates a random forest to predict whether traffic is benign or malicious
    :param x_train: The training data
    :param y_train: The training labels
    :param x_test: The test data
    :param y_test: The test labels
    :param trees: int: The number of trees in the classifier
    :param importance: bool: Whether to calculate feature importance scores
    :return: None
    """
    print(f"Creating a random forest with {trees} trees")
    clf = RandomForestClassifier(n_estimators=trees)
    clf.fit(x_train, y_train)
    print("Predicting Labels")
    y_pred = clf.predict(x_test)

    if importance:
        calculate_feature_importance(clf, x_test, y_test)

    print("Random Forest Metrics:")
    report_metrics(y_test, y_pred)


def calculate_feature_importance(clf, x_test, y_test, repeats=10, seed=123456, jobs=2):
    """
    Calculates feature importance scores based on the random forest classifier
    :param clf: The random forest
    :param x_test: The test data
    :param y_test: The test labels
    :param repeats: int: The amount of times to repeat for permutation importance
    :param seed: int: a random seed
    :param jobs: int: The number of jobs to use for permutation importance
    :return: None
    """

    print("Calculating fearture importance using mean decrease in impurity")
    importance_scores = clf.feature_importances_
    std = np.std([tree.feature_importances_ for tree in clf.estimators_], axis=0)
    feature_importance = pd.Series(importance_scores, index=clf.feature_names_in_)

    fig, ax = plt.subplots()
    feature_importance.plot.bar(yerr=std, ax=ax)
    ax.set_title("Feature importances using MDI")
    ax.set_ylabel("Mean decrease in impurity")
    fig.tight_layout()

    print("Calculating feature importance using permutation importance")
    importance_scores = permutation_importance(
        clf, x_test, y_test, n_repeats=repeats, random_state=seed, n_jobs=jobs
    )

    fig, ax = plt.subplots()
    feature_importance.plot.bar(yerr=importance_scores.importances_std, ax=ax)
    ax.set_title("Feature importances using permutation on full model")
    ax.set_ylabel("Mean accuracy decrease")
    fig.tight_layout()
    plt.show()


def tune_svm(x_train, y_train, x_test, y_test, sampling_ratio, mini, maxi, step, filename):
    """
    Creates and runs through multiple iterations of the svm model. Saves all output to a file.
    :param x_train: The training data
    :param y_train: The training labels
    :param x_test: The test data
    :param y_test: The test labels
    :param mini: Minimum of loop iterator, configure this for what parameter you want to test
    :param maxi: Maximum of loop iterator, configure for what parameter you want to test
    :param: step: Step value for testing loop
    :param: filename: The output file to save results
    :return: None
    """
    with open(filename, 'w+') as f_name:
        f_name.write('\n')
        f_name.write('Accuracy,Precision,Recall,F1-Score,AUC,True_Negative,False_Positive,'
                     'False_Negative,True_Positive\n')
        for i in range(mini, maxi, step):
            f_name.write(str(i))
            print(f"SVM iteration {i}")
            clf = svm.SVC(max_iter=i, kernel='rbf')
            clf.fit(x_train, y_train)
            y_pred = clf.predict(x_test)

            data = str(metrics.accuracy_score(y_test, y_pred)) + ','
            data += str(metrics.precision_score(y_test, y_pred)) + ','
            data += str(metrics.recall_score(y_test, y_pred)) + ','
            data += str(metrics.f1_score(y_test, y_pred)) + ','
            fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred)
            data += str(metrics.auc(fpr, tpr)) + ','
            tn, fp, fn, tp = metrics.confusion_matrix(y_test, y_pred).ravel()
            data += str(tn) + ','
            data += str(fp) + ','
            data += str(fn) + ','
            data += str(tp)
            f_name.write(data+'\n')


def tune_log_reg(x_train, y_train, x_test, y_test, sampling_ratio, mini, maxi, step, filename):
    """
    Creates and runs through multiple iterations of the logistic regression model. Saves all output to a file.
    :param x_train: The training data
    :param y_train: The training labels
    :param x_test: The test data
    :param y_test: The test labels
    :param mini: Minimum of loop iterator, configure this for what parameter you want to test
    :param maxi: Maximum of loop iterator, configure for what parameter you want to test
    :param: step: Step value for testing loop
    :param: filename: The output file to save results
    :return: None
    """
    with open(filename, 'w+') as f_name:
        f_name.write('\n')
        f_name.write('Accuracy,Precision,Recall,F1-Score,AUC,True_Negative,False_Positive,'
                     'False_Negative,True_Positive\n')
        for i in range(mini, maxi, step):
            f_name.write(str(i))
            print(f"LogR iteration {i}")

            clf = linear_model.LogisticRegression(C=i, random_state=0).fit(x_train, y_train)
            clf.fit(x_train, y_train)
            y_pred = clf.predict(x_test)
            data = str(metrics.accuracy_score(y_test, y_pred)) + ','
            data += str(metrics.precision_score(y_test, y_pred)) + ','
            data += str(metrics.recall_score(y_test, y_pred)) + ','
            data += str(metrics.f1_score(y_test, y_pred)) + ','
            fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred)
            data += str(metrics.auc(fpr, tpr)) + ','
            tn, fp, fn, tp = metrics.confusion_matrix(y_test, y_pred).ravel()
            data += str(tn) + ','
            data += str(fp) + ','
            data += str(fn) + ','
            data += str(tp)
            f_name.write(data+'\n')


def tune_neural(x_train, y_train, x_test, y_test, sampling_ratio, mini, maxi, step, filename):
    """
    Creates and runs through multiple iterations of the neural network model. Saves all output to a file.
    :param x_train: The training data
    :param y_train: The training labels
    :param x_test: The test data
    :param y_test: The test labels
    :param mini: Minimum of loop iterator, configure this for what parameter you want to test
    :param maxi: Maximum of loop iterator, configure for what parameter you want to test
    :param: step: Step value for testing loop
    :param: filename: The output file to save results
    :return: None
    """
    with open(filename, 'w+') as f_name:
        f_name.write('\n')
        f_name.write('Accuracy,Precision,Recall,F1-Score,AUC,True_Negative,False_Positive,'
                     'False_Negative,True_Positive\n')
        for i in range(mini, maxi, step):
            f_name.write(str(i))
            print(f"NeuralNetwork: iteration {i}")

            clf = MLPClassifier(max_iter=i, solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
            clf.fit(x_train, y_train)
            y_pred = clf.predict(x_test)
            data = str(metrics.accuracy_score(y_test, y_pred)) + ','
            data += str(metrics.precision_score(y_test, y_pred)) + ','
            data += str(metrics.recall_score(y_test, y_pred)) + ','
            data += str(metrics.f1_score(y_test, y_pred)) + ','
            fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred)
            data += str(metrics.auc(fpr, tpr)) + ','
            tn, fp, fn, tp = metrics.confusion_matrix(y_test, y_pred).ravel()

            data += str(tn) + ','
            data += str(fp) + ','
            data += str(fn) + ','
            data += str(tp)
            f_name.write(data+'\n')


def logistic_regression(x_train, y_train, x_test, y_test, max_iters=5000):
    """
    Uses logistic regression to predict the class of the traffic
    :param x_train: The training data
    :param y_train: The training labels
    :param x_test: The test data
    :param y_test: The test labels
    :param max_iters: int: The maximum iterations to use for the classifier
    :return: None
    """
    print("Creating a Logistic Regression model to classify the traffic")
    clf = linear_model.LogisticRegression(max_iter=max_iters)
    clf.fit(x_train, y_train)
    print("Predicting labels")
    y_pred = clf.predict(x_test)
    print("Logistic Regression Metrics:")
    report_metrics(y_test, y_pred)


def neural_network(x_train, y_train, x_test, y_test, solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), seed=123456,
                   max_iters=5000):
    """
    Uses a neural network to classify the data
    :param x_train: The training data
    :param y_train: The training labels
    :param x_test: The testing data
    :param y_test: The testing labels
    :param solver: str: The method used for back propagation
    :param alpha: int: The learning rate of the network
    :param hidden_layer_sizes: tuple: The number of hidden layers and neurons per layer
    :param seed: int: A random seed
    :param max_iters: int: The maximum iterations to train the model
    :return: None
    """
    print("Creating a Neural Network to classify traffic")
    clf = MLPClassifier(solver=solver, alpha=alpha, hidden_layer_sizes=hidden_layer_sizes, random_state=seed,
                        max_iter=max_iters)
    clf.fit(x_train, y_train)
    print("Predicting labels")
    y_pred = clf.predict(x_test)
    print("Neural Network Metrics:")
    report_metrics(y_test, y_pred)


def svm_classifier(x_train, y_train, x_test, y_test, kernel='rbf', max_iters=5000):
    """
    Uses an SVM to classify the traffic
    :param x_train: The training data
    :param y_train: The training labels
    :param x_test: The test data
    :param y_test: The testing labels
    :param kernel: str: The kernel function to use
    :param max_iters: int: The max iterations used to train the classifier
    :return: None
    """
    print(f"Creating an SVM with a {kernel} kernel to classify the traffic")
    clf = svm.SVC(kernel=kernel, max_iter=max_iters)
    clf.fit(x_train, y_train)
    print("Predicting labels")
    y_pred = clf.predict(x_test)
    print("SVM Metrics:")
    report_metrics(y_test, y_pred)


def main():
    x_train, x_test, y_train, y_test = prepare_data('capture20110810.h5', 10, ['window_id'], 6)
    random_forest(x_train, y_train, x_test, y_test)
    logistic_regression(x_train, y_train, x_test, y_test, max_iters=10000)
    neural_network(x_train, y_train, x_test, y_test, max_iters=10000)
    svm_classifier(x_train, y_train, x_test, y_test, max_iters=10000)


if __name__ == '__main__':
    main()
