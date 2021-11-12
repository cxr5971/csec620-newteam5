import numpy
import math
import matplotlib.pyplot as plots

from sklearn.decomposition import PCA


def euclidean_distance(point_one: numpy.ndarray, point_two: numpy.ndarray):
    """
    A simple implementation of the Euclidean distance algorithm
    :param point_one: numpy.ndarray: The first point
    :param point_two: numpy.ndarray: The second point
    :return: The calculated distance between the two points
    """
    for i in range(len(point_one)):
        total = (point_one[i] - point_two[i])**2
    return math.sqrt(total)


def load_data_file(filename: str):
    """
    A simply function to load the numpy data. Made in case we want to load in another file but still want all data
    processing here
    :param filename: str: The name of the file containing our numpy data
    :return: A numpy.ndarray of the loaded data
    """
    return numpy.load(filename)


def train_pca(training_data: numpy.ndarray, num_of_components: int):
    """
    A function that creates the PCA model and trains it based on the provided data
    :param training_data: numpy.ndarray: The data used to train the PCA model
    :param num_of_components: int: the number of components/features to compress the data into
    :return: The trained PCA model
    """
    pca_model = PCA(num_of_components)
    pca_model.fit(training_data)
    return pca_model


def prepare_data(n_components: int=2):
    """
    This function loads in the training and test data (currently hardcoded file paths) and creates the PCA model based
    on the training data. Once the model is trained we use it to reduce the dimensions in our test data
    :param n_components: int: the number of components to condense the data into. Default = 2
    :return: The reduced numpy.ndarray testing sets
    """
    training_data = numpy.load("KDD99/training_normal.npy")
    testing_normal = numpy.load("KDD99/testing_normal.npy")
    testing_malicious = numpy.load("KDD99/testing_attack.npy")

    pca_model = PCA(n_components=n_components)
    pca_model.fit(training_data)

    reduced_training_data = pca_model.transform(training_data)
    reduced_test_normal = pca_model.transform(testing_normal)
    reduced_test_malicious = pca_model.transform(testing_malicious)

    return reduced_test_normal, reduced_test_malicious, reduced_training_data


def make_subset(numpy_array: numpy.ndarray, sample_size: int):
    """
    This function creates a subset of our data for the scatter plot
    :param numpy_array: numpy.ndarray: The data to take a subset from
    :param sample_size: int: The size of our resulting subset
    :return: The subset of our data
    """
    subset = numpy_array[numpy.random.randint(numpy_array.shape[0], size=sample_size), :]
    return subset


def make_scatter_plot(normal_data: numpy.ndarray, malicious_data: numpy.ndarray):
    """
    Uses matplotlip to make our scatter plot of the normal and malicious test data
    :param normal_data: numpy.ndarray: The subset of our normal data
    :param malicious_data: numpy.ndarry: The subset of our malicious data
    :return: None
    """
    scheming = plots.figure()
    x_axis = scheming.add_subplot(111)
    x_axis.scatter(normal_data[:, 0], normal_data[:, 1], c="b", marker="o", label="Normal")
    x_axis.scatter(malicious_data[:, 0], malicious_data[:, 1], c="r", marker="^", label="Malicious")
    plots.xlabel("PC1")
    plots.ylabel("PC2")
    plots.legend(loc="upper left")
    plots.title("Scatter plot of reduced normal and malicious test data")
    plots.show()


def main():
    """
    Makes the scatter plot of the data
    :return:
    """
    reduced_test_normal, reduced_test_malicious, reduced_training = prepare_data(2)
    subset_test_normal = make_subset(reduced_test_normal, 500)
    subset_test_malicious = make_subset(reduced_test_malicious, 500)

    make_scatter_plot(subset_test_normal, subset_test_malicious)


if __name__ == '__main__':
    main()
