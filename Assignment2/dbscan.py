import process_data
import numpy
from matplotlib import pyplot
from sklearn.neighbors import NearestNeighbors


def is_core_point(point_index: int, training_data: numpy.ndarray, epsilon: int, min_neighbors: int):
    """
    This method determines whether a given point is a core point. It loops over all the points in the data and checks if
    the distance between itself and the point is less than epsilon. If it is we add 1 to our neighbor count. If the
    count is larger than the minimum neighbors, we return True other wise we keep looping.
    :param point_index: int: The index of the point being classified
    :param training_data: numpy.ndarray: The numpy array of training points
    :param epsilon: int: The maximum distance a point can be and still be a neighbor
    :param min_neighbors: int: The minimum number of neighbors needed to be a core point
    :return: True if a core point, false otherwise
    """
    point = training_data[point_index]
    neighbor_count = 0

    for i in range(training_data.shape[0]):
        if i == point_index:  # I can be my own friend but not my own neighbor
            continue

        elif numpy.linalg.norm(point - training_data[i]) <= epsilon:  # Good old Euclid
            neighbor_count += 1

            if neighbor_count == min_neighbors:  # If we know we are a core point we can stop
                return True

    return False


def find_core_points(training_data: numpy.ndarray, epsilon: int, min_neighbors: int):
    """
    Loops over all the points in the data set searching for core points
    :param training_data: numpy.ndarray: All the points in the training data
    :param epsilon: int: The maximum distance a point can be and still be a neighbor
    :param min_neighbors: int: The minimum number of neighbors required to be a core point
    :return: A list of indexes for all the core points
    """
    core_point_indexes = []
    for index in range(training_data.shape[0]):
        if is_core_point(index, training_data, epsilon, min_neighbors):
            core_point_indexes.append(index)
    return core_point_indexes


def predict_data(core_points: list, training_data: numpy.ndarray, test_data: numpy.ndarray, epsilon: int):
    """
    This method determines whether our testing data is part of a cluster. For each test sample, we loop over all the
    core points and check if the distance between the core point and test point is less than or equal to epsilon. If it
    is, we know the test point belongs to a cluster so we add it to the normal data list (good_boys) and move on to the
    next point. If the test point isn't neighbors with any core points its anomaly and we judge it harshly.
    :param core_points: list(int): A list containing the index of every core point
    :param training_data: numpy.ndarray: The training points used to build the clusters
    :param test_data: numpy.ndarray: The test data being predicted
    :param epsilon: int: The maximum distance away from a point a second point can be and still be its neighbor
    :return: Two lists; one containing all the normal data (good_boys) and one containing all the anomalies
    """
    good_boys = []
    anomalies = []

    for test_sample in test_data:
        anomaly = True  # A sentinel to prevent adding the sample to both lists

        """
        essentially we a treating the data as clustered even though we didn't formally build the clusters. Since we 
        only care whether the point is in a cluster and not which cluster, there is no benefit to knowing which cluster
        the point is in. Instead we use the idea of a cluster to our advantage. To be in a cluster you must be a core 
        points neighbor, so if a test point is neighbors with any core point we know it is in a and thus not an anomaly
        """
        for index in core_points:
            core_point = training_data[index]
            if numpy.linalg.norm(test_sample - core_point) <= epsilon:  # In a cluster so we are normal
                good_boys.append(test_sample)
                anomaly = False
                break

        # If a point is not a neighbor with any core point its an anomaly
        if anomaly:
            anomalies.append(test_sample)

    return good_boys, anomalies


def report_results(true_positives: list, true_negatives: list, false_positives: list, false_negatives: list):
    """
    This method handles all of our reporting
    :param true_positives: list(numpy.ndarray): A list of all the anomalous data marked anomalous
    :param true_negatives: list(numpy.ndarray): A list of all the normal data marked normal
    :param false_positives: list(numpy.ndarry): A list of all the normal data marked anomalous
    :param false_negatives: list(numpy.ndarray): A list of all the anomalous data marked normal
    :return: None
    """
    tp = len(true_positives)
    tn = len(true_negatives)
    fp = len(false_positives)
    fn = len(false_negatives)

    accuracy = (tp + tn) / (tp + fp + tn + fn)
    true_positive_rate = tp / (fn + tp)
    false_positive_rate = fp / (tn + fp)
    f1_score = (2 * tp) / (2 * tp + fp + fn)

    print(f"Accuracy: {accuracy}\nTrue Positive Rate: {true_positive_rate}\nFalse Positive Rate: {false_positive_rate}"
          f"\nF1 Score: {f1_score}")


def k_nearest_neighbors(k: int, training_data: numpy.ndarray):
    """
    This is a modified form of our Knn algorithm from assignment1. This method finds each points k nearest neighbors
    and saves the distances in a list. It then adds these lists of distances to another list. We needed this method
    to tune the hyper-parameters for our training data with 41 features. We kept getting a memory error when using
    sklearn's NearestNeighbor method
    :param k: int: The number of nearest neighbors to calculate for each point
    :param training_data: numpy.ndarray: All the training points
    :return: A 2d array. A list of lists of distances representing the distance from each point to its k nearest
             neighbors
    """
    neighbor_distances = []

    for index in range(training_data.shape[0]):
        print(index)
        distances = []
        for second_index in range(training_data.shape[0]):
            distances.append(numpy.linalg.norm(training_data[index] - training_data[second_index]))

        distances.sort()
        nearest_neighbors = distances[:k]

        neighbor_distances.append(nearest_neighbors)
    return neighbor_distances


def compute_averages(min_neighbors: int, neighbors: list):
    """
    A simple function to take the first x number of data points in a list and calculate the average. This allows us
    to only need to do Knn once but compute the optimization graph for multiple min_neighbors
    :param min_neighbors: int: The minimum number of neighbors choice being tested
    :param neighbors: list(list(int)): The 2D array returned by Knn
    :return: A list of the average distance from a point to its neighbors for every point
    """
    averages = []
    for sample in neighbors:
        averages.append(sum(sample[:min_neighbors])/min_neighbors)
    return averages


def hype_train(training_data: numpy.ndarray):
    """
    A function used to create a graph for our hyper parameter training. This method uses our Knn algorithm to find the
    distances between each point and its 20 nearest neighbors. Then it loops over 2 through 19 neighbors, computes the
    average distance form each point to its neighbors, and plots the results. From these plots we can choose our optimal
    epsilon for a given number of neighbors by looking at the elbow in the graph
    :param training_data: numpy.ndarray: The training points to optimize
    :return: None
    """
    neigbor_distances = k_nearest_neighbors(20, training_data)
    for min_neigbors in range(2, 20):
        distances = compute_averages(min_neigbors, neigbor_distances)
        distances.sort()
        pyplot.title(f"{min_neigbors} neighbors to be a core point")
        pyplot.plot(distances)
        pyplot.show()


def hype_train_two(min_min: int, max_min: int, training_data: numpy.ndarray):
    """
    A faster implementation of our hyper-parameter tuner. This tuner uses sklearn's NearestNeighbor method for finding
    the distances to the k neareast neighbors. It is much faster than our implementation, but can not handle the data
    if it has many features (at least 20+). Then it plots the distances so we can again find the optimal value for
    epsilon given a minimum number of neighbors. This function also allows for more fine tuned control of the
    min_neighbors being tested
    :param min_min: int: The start value for our range
    :param max_min: int: The end value for our range
    :param training_data: numpy.ndarray: The training data used to train the algorithm
    :return: None
    """
    for i in range(min_min, max_min):  # The new and IMPROVED loop
        neighbors = NearestNeighbors(n_neighbors=i)
        fir_neighbors = neighbors.fit(training_data)
        distances, indices = fir_neighbors.kneighbors(training_data)
        distances = numpy.sort(distances, axis=0)
        pyplot.title(f"{i} neighbors to be a core point")
        pyplot.plot(distances[:, 1])
        pyplot.show()


def main():
    num_features = 41
    min_neighbors = 5
    epsilon = .06

    # load the data
    reduced_testing_normal, reduced_testing_malicious, reduced_training_data = process_data.prepare_data(num_features)

    # train and predict
    core_points = find_core_points(reduced_training_data, epsilon, min_neighbors)
    true_negatives, false_positives = predict_data(core_points, reduced_training_data, reduced_testing_normal, epsilon)
    false_negatives, true_positives = predict_data(core_points, reduced_training_data, reduced_testing_malicious,
                                                   epsilon)

    # report
    print(f"Results for DBSCAN using {num_features} features, {min_neighbors} minimum neighbors, and {epsilon} "
          f"for epsilon")
    report_results(true_positives, true_negatives, false_positives, false_negatives)


if __name__ == "__main__":
    main()
