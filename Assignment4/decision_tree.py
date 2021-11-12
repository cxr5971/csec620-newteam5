import csv
import random


class SplitNode:
    """
    A class to represent a split in a decision tree
    Fields:
        right_node: Union[SplitNode, LeafNode]: The right branch of the tree (if the condition is true)
        left_node: Union[SplitNode, LeafNode]: The left branch of the tree (if the condition is false)
        impurity: int: The impurity value of the split
        feature_index: int: The feature in the sample used for the split
        split_value: int: The value to compare the feature against
    """

    def __init__(self, impurity, feature_index, split_value):
        self.right_node = None
        self.left_node = None
        self.impurity = impurity
        self.feature_index = feature_index
        self.split_value = split_value


class LeafNode:
    """
    A class to represent an classification in a decision tree
    Fields:
        samples: list: all the samples that belong to the node
        prediction: int: The most common device type in the node
        impurity: int: The gini impurity for the leaf node (just cuz)
    """

    def __init__(self, samples, impurity, prediction):
        self.samples = samples
        self.prediction = prediction
        self.impurity = impurity


def decision_tree_training(max_depth, min_samples, x_features, x_labels, feature_subcount=12):
    """
    This method builds the decision tree. Honestly it was used for testing functionality before everything was built,
    but now it simply just calls build_tree
    :param max_depth: int: The max depth any branch in the tree can reach
    :param min_samples: int: The minimum number of samples a node must have to split again
    :param x_features: numpy.ndarray: The training samples
    :param x_labels: numpy.ndarry: The training labels
    :param: feature_subcount: int: The number of features to use at each split. Defaults to 12
    :return: The root node of the tree
    """

    tree = build_tree(max_depth, min_samples, x_features, x_labels, 0, 1, feature_subcount)
    return tree


def random_forest_training(n_trees, data_frac, feature_subset, max_depth, min_samples, samples, labels):
    trees = []
    for i in range(0, n_trees):
        data_subset = []
        label_subset = []
        for count in range(round(data_frac * len(samples))):
            sample_index = random.randint(0, len(samples) - 1)
            data_subset.append(samples[sample_index])
            label_subset.append(labels[sample_index])
        tree = decision_tree_training(max_depth, min_samples, data_subset, label_subset, feature_subset)
        trees.append(tree)
    return trees


def make_leaf_node(samples, labels):
    """
    This function simply makes a LeafNode. It calculates the node's impurity and the class used to classify samples
    in this node
    :param samples: list or maybe a numpy.ndarry: The samples in the LeafNode
    :param labels: list or numpy.ndarray: The labels of the samples in the LeafNode
    :return: A LeafNode
    """
    impurity = gini_impurity_leaf_node(labels)  # Not important now but I think it might be for the report
    prediction = mode(labels)
    return LeafNode(samples, impurity, prediction)


def build_tree(max_depth, min_node, samples, labels, depth, parent_impurity, feature_subcount):
    """
    A recursive function that finds the optimal split and builds the decision tree
    :param max_depth: int: The max recursion depth
    :param min_node: int: The minimum number of samples needed to keep splitting a node
    :param samples: list or numpy.ndarray: The samples for the current node
    :param labels: list or numpy.ndarray: The labels for the current node
    :param depth: int: The current depth in the tree
    :param parent_impurity: int: The impurity of the previous node
    :param: feature_subcount: int: The number of features to check per split
    :return: The next node in the tree; SplitNode or LeafNode
    """
    if depth > max_depth:  # Base case 1: max recursion reached
        return make_leaf_node(samples, labels)
    elif len(samples) < min_node:  # Base case 2: Not enough samples to continue
        return make_leaf_node(samples, labels)
    else:  # Otherwise find the best split
        split_info = optimal_split(samples, labels, feature_subcount)
        if split_info[0] > parent_impurity:  # Base case (kinda) 3: No better split of the data
            return make_leaf_node(samples, labels)
        # Base case 4: Not really a split if only 1 side has data. Also probably never reached thanks to case 3
        elif len(split_info[3]) == 0 or len(split_info[5]) == 0:
            return make_leaf_node(samples, labels)
        else:  # If all else fails lets split the data
            split_node = SplitNode(split_info[0], split_info[1], split_info[2])
            split_node.left_node = build_tree(max_depth, min_node, split_info[3], split_info[4],
                                              depth+1, split_info[0], feature_subcount)
            split_node.right_node = build_tree(max_depth, min_node, split_info[5], split_info[6],
                                               depth + 1, split_info[0], feature_subcount)
            return split_node


# Thought about combining with the impurity function to remove redundant loops but saw no change to performance
def mode(labels):
    """
    A function that determines the most common class in a leaf node and returns the value. Used to figure out what
    each leaf node should use for its classification
    :param labels: list or numpy.ndarray: The labels of all the samples for the given leaf node
    :return: The class that occurs the most out of all the samples at the node
    """
    class_dict = {}

    # Determine the count of each class for the node
    for label in labels:
        if label in class_dict.keys():
            class_dict[label] += 1
        else:
            class_dict[label] = 1

    # Find the highest count
    max_count = 0
    max_class = None
    for key, value in class_dict.items():
        if value > max_count:
            max_count = value
            max_class = key

    return max_class


def feature_splits(feature, samples):
    """
    A helper method for optimal splits. Given a selected feature it finds all the unique values it can be in the sample
    data. It then returns a sorted list of these values
    :param feature: int: The index of the feature we are checking
    :param samples: list or numpy.ndarray: The samples at the current node
    :return: A sorted list of unique feature values
    """
    # Get unique features
    feature_values = set()
    for sample in samples:
        feature_values.add(sample[feature])

    # Sort them
    feature_values = list(feature_values)
    feature_values.sort()
    return feature_values


def optimal_split(samples, labels, features=12):
    """
    The core of the decision tree. This loops over all the features of all the samples for the current node and uses
    gini impurity to find the optimal split. While doing this it not only finds the best split but also performs the
    split
    :param samples: list or numpy.ndarray: The samples at the current node
    :param labels: list or numpy.ndarray: The labels of the samples at the current node
    :param: features: int: The number of features to check per split. Default 12
    :return: All information needed to split the data
        best_impurity: int: The gini impurity of the best split
        best_feature: int: The index of the feature used to make the split
        best_value: int or float: The value used for the best split
        best_conditional: str: A character representing the conditional operation to use (<, =, >)
        best_left_samples: list: The samples that make up the left (False) side of the split
        best_left_labels: list: The labels of the best_left_samples samples
        best_right_samples: list: The samples in the right (True) side of the split
        best_right_labels: list: The labels for the right side samples
    """

    #  Creating many variables to hold the best know split information
    best_impurity = 1
    best_feature = None
    best_value = None
    best_left_samples = []
    best_left_labels = []
    best_right_labels = []
    best_right_samples = []

    list_of_features = []
    if features == 12:
        list_of_features = [x for x in range(12)]
    else:
        for count in range(0, features + 1):
            random_feature_index = random.randint(0, features - 1)
            list_of_features.append(random_feature_index)

    for feature in list_of_features:  # for all the features in the samples
        feature_values = feature_splits(feature, samples)  # Get all possible values of the feature

        for value in feature_values:  # for each possible value
            left = []
            left_labels = []
            right = []
            right_labels = []
            for i in range(len(samples)):
                # Originally we checked <, ==, and > but we found that simply checking <= works best for a single tree
                if samples[i][feature] <= value:
                    right.append(samples[i])
                    right_labels.append(labels[i])
                else:
                    left.append(samples[i])
                    left_labels.append(labels[i])

            impurity = total_gini_impurity(left_labels, right_labels)  # calculate the impurity of the split

            if impurity < best_impurity:  # if the impurity is lower, update the best values
                best_impurity = impurity
                best_feature = feature
                best_value = value
                best_left_samples = left[:]
                best_left_labels = left_labels[:]
                best_right_labels = right_labels[:]
                best_right_samples = right[:]

    # Finally a data dump of information
    return best_impurity, best_feature, best_value, best_left_samples, \
        best_left_labels, best_right_samples, best_right_labels


def gini_impurity_leaf_node(node_labels):
    """
    Calculates the gini impurity of a leaf node (one side of the split)
    :param node_labels: list or numpy.ndarray: The labels of the samples in the leafnode
    :return: The impurity of the leaf node
    """
    count_dict = {}
    total_samples = len(node_labels)

    # count each class
    for label in node_labels:
        if label in count_dict:
            count_dict[label] += 1
        else:
            count_dict[label] = 1

    # calculate impurity
    impurity = 1
    for class_count in count_dict.values():
        probability = class_count/total_samples
        impurity -= probability**2
    return impurity


def total_gini_impurity(left_node_labels, right_node_labels):
    """
    Calculates the gini impurity of the given split i.e. the weighted average of the left and right node after the
    split
    :param left_node_labels: list or numpy.ndarray: The labels of all the samples in the right split
    :param right_node_labels: list or numpy.ndarray: The labels of all the samples in the left split
    :return: The overall impurity of the split
    """
    total_samples = len(left_node_labels) + len(right_node_labels)
    left_impurity = gini_impurity_leaf_node(left_node_labels)
    right_impurity = gini_impurity_leaf_node(right_node_labels)
    total_impurity = ((len(left_node_labels)/total_samples) * left_impurity) + \
                     ((len(right_node_labels)/total_samples) * right_impurity)
    return total_impurity


def predict(tree, sample):
    """
    An iterative method the traverse the tree and predict the class of a new sample
    :param tree: SplitNode: The root node of the tree to traverse
    :param sample: numpy.ndarray: The sample to classify
    :return: The predicted class of the sample
    """
    current_node = tree
    while not isinstance(current_node, LeafNode):  # If we are at a leaf node, we cant go any farther
        feature = current_node.feature_index
        value = current_node.split_value

        if sample[feature] <= value:
            current_node = current_node.right_node
        else:
            current_node = current_node.left_node

    return current_node.prediction


def dt_accuracy(tree, test_set, test_labels):
    """
    This function loops over all the test data and predicts their classes. The function then counts the number of
    correct and incorrect predictions and reports the accuracy of the model
    :param tree: SplitNode: The root node of the decision tree
    :param test_set: numpy.ndarray: The test samples
    :param test_labels: numpy.ndarray: The labels of the test data
    :return: None
    """
    correct = wrong = 0
    all_predictions = []
    for i in range(len(test_set)):
        sample = test_set[i]
        label = test_labels[i]
        prediction = predict(tree, sample)

        if label == prediction:
            correct += 1
        else:
            wrong += 1
        all_predictions.append(prediction)
    print(f"correct: {correct}")
    print(f"wrong: {wrong}")
    print(f"accuracy: {correct/ (correct+wrong)}")
    return all_predictions


def rf_accuracy(trees, samples, labels):
    sample_predictions = {}
    for tree in trees:
        sample_index = 0
        for sample in samples:
            prediction = predict(tree, sample)
            if sample_index not in sample_predictions:
                sample_predictions[sample_index] = {}
            if prediction in sample_predictions[sample_index]:
                sample_predictions[sample_index][prediction] += 1
            else:
                sample_predictions[sample_index][prediction] = 1
            sample_index += 1

    final_predictions = []
    correct = wrong = 0
    for sample_index in sample_predictions.keys():
        final_prediction = max(sample_predictions[sample_index], key=sample_predictions[sample_index].get)
        final_predictions.append(final_prediction)
        if labels[sample_index] == final_prediction:
            correct += 1
        else:
            wrong += 1

    print("Correct: " + str(correct))
    print("Wrong: " + str(wrong))
    print("Accuracy: " + str(correct / (correct + wrong)))
    return final_predictions


def make_confusion_matrix(predictions, labels, classes, count=""):
    """
    Builds a confusion matrix and normalized confusion matrix for the predictions
    :param predictions: list: The predicted values of all the samples in the test set
    :param labels: numpy.ndarray: All the actual labels of the test data
    :param classes: numpy.ndarray: The MAC addresses that corespond to each class
    :param count: str: A string used to change the filename if performing multiple runs
    :return: None
    """
    # Lets build an empty dict
    confusion_dict = {}
    for i in range(24):
        confusion_dict[i] = [0] * 24

    # For each sample we count each class it was classified as
    for i in range(len(labels)):
        prediction = predictions[i]
        label = labels[i]
        confusion_dict[label][prediction] += 1

    confusion_core = list(confusion_dict.values())

    # FILES
    raw_matrix_file = open(f"confusion_matrix{count}.csv", mode="w", newline="")
    normalized_matrix_file = open(f"normalized_confusion_matrix{count}.csv", mode="w", newline="")

    raw_matrix_writer = csv.writer(raw_matrix_file, delimiter=",", quotechar='"')
    normalized_matrix_writer = csv.writer(normalized_matrix_file, delimiter=",", quotechar='"')

    # Adding None to make the column headers line up right
    raw_matrix_writer.writerow([None] + list(classes))
    normalized_matrix_writer.writerow([None] + list(classes))
    for i in range(24):
        row = confusion_core[i]
        normalized_row = row[:]

        # For the raw case we simple just write out the list
        row.insert(0, classes[i])
        raw_matrix_writer.writerow(row)

        # Else we get the total number of samples for a given class and divide each count by the total
        total_samples = sum(normalized_row)
        for j in range(24):
            normalized_row[j] = round(normalized_row[j] / total_samples,3)
        normalized_row.insert(0, classes[i])
        normalized_matrix_writer.writerow(normalized_row)

    # Never forget the cleanup
    raw_matrix_file.close()
    normalized_matrix_file.close()
