from decisiontree import DecisionTreeClassifier
import random
from collections import defaultdict

class RandomForestClassifier:
    #TODO: as an optimizer maybe thread all of the trees?
    #TODO: Could multiprocess this? the downsides of python
    def __init__(self, n_trees, data_frac, feature_subcount, max_depth, min_node):
        self.n_trees = n_trees
        self.data_frac = data_frac
        self.feature_subcount = feature_subcount
        self.max_depth = max_depth
        self.min_node = min_node

    def fit(self, samples, labels):
        #train the model
        self.trees = list()
        subset_size = round((self.data_frac * len(samples)))
        for _ in range(self.n_trees):
            subset_samples = list()
            subset_labels = list()
            for __ in range(subset_size):
                index = random.randint(0, len(samples) - 1)
                subset_samples.append(samples[index])
                subset_labels.append(labels[index])
            tree = DecisionTreeClassifier(self.max_depth, self.min_node, self.feature_subcount)
            tree.build_tree(subset_samples, subset_labels)
            self.trees.append(tree)

    def predict(self, sample):
        results = defaultdict(lambda: 0)
        for tree in self.trees:
            prediction = tree.predict(sample)
            results[prediction] = results[prediction] + 1
        best_prediction = 0
        best_count = 0
        #count the votes
        for prediction in results:
            if results[prediction] > best_count:
                best_count = results[prediction]
                best_prediction = prediction
        return best_prediction
    
    def predict_multiple_samples(self, samples):
        results = list()
        for sample in samples:
            results.append(self.predict(sample))
        return results
