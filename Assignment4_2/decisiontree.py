from collections import defaultdict
import statistics
import random

class Node:
    def is_leaf(self):
        return False

class DecisionNode(Node):
    def __init__(self, index, value, left, right):
        self.index = index
        self.value = value
        self.left = left
        self.right = right

    def is_leaf(self):
        return False

class LeafNode(Node):

    def __init__(self, samples, labels):
        self.samples = samples
        self.labels = labels
    
    def predict(self):
        return statistics.mode(self.labels)
    
    def is_leaf(self):
        return True

class DecisionTreeClassifier:
    max_depth = -1
    min_node = -1
    root_node = None
    def __init__(self, max_depth, min_node, feature_subcount=-1):
        self.max_depth = max_depth
        self.min_node = min_node
        self.feature_subcount = feature_subcount
    
    def build_tree(self, samples, labels):
        self.root_node = self.build(samples, labels)
         
    def build(self, samples, labels, depth=0, p_impurity=-1):
        if self.max_depth != -1 and depth > self.max_depth:
            return LeafNode(samples, labels)
        elif len(samples) < self.min_node:
            return LeafNode(samples, labels)
        else:
            best_impurity, best_feature, best_value, best_left, best_right = self.split_by_optimal_feature(samples, labels, self.feature_subcount)
            samples_left = best_left[0]
            samples_right = best_right[0]
            labels_left = best_left[1]
            labels_right = best_right[1]
            if len(samples_left) == 0 or len(samples_right) == 0:
                return LeafNode(samples, labels)
            elif p_impurity != -1 and best_impurity < p_impurity:
                return LeafNode(samples, labels)
            else:
                split = DecisionNode(best_feature, best_value, \
                    self.build(samples_left, labels_left, depth=depth+1, p_impurity=best_impurity),\
                    self.build(samples_right, labels_right, depth=depth+1, p_impurity=best_impurity))
                return split

    def split_by_optimal_feature(self, samples, labels, features):
        features_to_check = list()
        
        features_to_check.extend(range(len(samples[0])))
        if features != -1:
            temp = random.choices(features_to_check, k=features)
            features_to_check = temp
        best_left = None
        best_right = None
        best_feature = None
        best_value = None
        best_impurity = 1
        for index in features_to_check:
            # get all the feature values

            feature_values = list()
            for sample in samples:
                feature_values.append(sample[index])
            for value in feature_values:
                left_samples = list()
                right_samples = list()
                left_labels = list()
                right_labels = list()
                for i in range(len(samples)):
                    #TODO: Account for equals case better than this
                    if samples[i][index] <= value:
                        left_samples.append(samples[i])
                        left_labels.append(labels[i])
                    else:
                        right_samples.append(samples[i])
                        right_labels.append(labels[i])
                impurity = self.calc_gini_impurity(left_labels, right_labels)
                if impurity < best_impurity: #TODO: should this be greater or lesser
                    best_impurity = impurity
                    best_feature = index
                    best_value = value
                    best_left = (left_samples, left_labels)
                    best_right = (right_samples, right_labels)

        return best_impurity, best_feature, best_value, best_left, best_right
        

    def predict(self, sample):
        #take a sample and predict.
        #continuously walk down the node
        node = self.root_node
        #print(self.root_node.is_leaf())
        while not node.is_leaf():
            if sample[node.index] <= node.value:
                node = node.left
            else:
                node = node.right
        return node.predict()

    """def calc_gini_impurity(self, left_labels, right_labels):
        left_counts = defaultdict(lambda: 0)
        right_counts = defaultdict(lambda: 0)
        for label in left_labels:
            left_counts[label] = left_counts[label] + 1
        for label in right_labels:
            right_counts[label] = right_counts[label] + 1
        
        left_impurity = self.__calc_gini_impurity(left_counts)
        right_impurity = self.__calc_gini_impurity(right_counts)
        total = len(left_labels) + len(right_labels)
        return ((len(left_labels) / total) * left_impurity) + ((len(right_labels) / total) * right_impurity)
    

    def __calc_gini_impurity(self, value_counts):
        counter = 0
        n = sum(value_counts)
        for key in value_counts:
            counter = counter + ((value_counts[key] / n) * (value_counts[key] / n))
        return 1 - counter"""
    
    def calc_gini_impurity(self, left_labels, right_labels):
        #this is going to get filtered out eventually as a leaf node anyway.
        #might as well make this the worst split possible.
        if len(left_labels) == 0 or len(right_labels) == 0:
            return 1
        total_labels = len(left_labels) + len(right_labels)
        left = self.__calc_gini_impurity(left_labels)
        right = self.__calc_gini_impurity(right_labels)
        total = (len(left_labels) / total_labels) * left
        total += (len(right_labels) / total_labels) * right
        return total
        

    def __calc_gini_impurity(self, labels):
        total = len(labels)
        counts = defaultdict(lambda: 0)
        for label in labels:
            counts[label] = counts[label] + 1
        if total == 0:
            print('WARNING: DIVIDE BY ZERO ABOUT TO HAPPEN')
        result = 1
        #for each class, take probability ^2  and minus from impurity
        for count in counts.values():
            prob = count / total
            result -= pow(prob, 2)
        return result
