import random
import zipfile


def load_sample_data(directory_name: str, labels: dict):
    """
    This function opens the zipped directory and loads each sample from the directory. It also separates the samples by
    benign and malicious.
    :param directory_name: str: The name of the zipped directory
    :param labels: dict: A dictionary containing the ids of all the malicious samples
    :return: All the samples and their features split between a malicious and a benign dictionary
    """
    benign_samples = {}
    malware_samples = {}

    with zipfile.ZipFile(directory_name, "r") as data_dir:  # opens zipped dir without having to actually extract it
        for sample in data_dir.namelist():  # loop over each sample
            sample_name = sample.split("/")[1]
            sample_features = set()

            for line in data_dir.open(sample):  # load each sample's features
                line = line.decode('utf-8')
                line = line.strip()
                sample_features.add(line)

            if sample_name in labels.keys():
                malware_samples[sample_name] = sample_features
            else:
                benign_samples[sample_name] = sample_features

    return benign_samples, malware_samples


def load_label_data(filename: str):
    """
    Creates two dictionaries. One maps a malware sample to the type of malware it is while the other maps a malware type
    to all the samples of that type in the dataset
    :param filename: str: The file of all the labeled data
    :return: A dictionary of samples and their labels and a dictionary of labels and their malware
    """
    labels = dict()
    malware_families = dict()

    with open(filename) as csv_file:
        next(csv_file)

        for record in csv_file:
            record = record.split(",")
            label = record[1].strip()
            sample = record[0]
            labels[sample] = label  # sampleID -> label
            if label in malware_families.keys():
                malware_families[label].append(sample)  # label -> list of samples
            else:
                malware_families[label] = [sample]
        return labels, malware_families


def create_training_and_testing_sets_ova(malware_samples: dict, training_ratio: int, malware_families: dict):
    """
    This function builds a training and testing data set from the overall data set. The function under samples the
    benign samples according to the benign ratio.
    :param benign_samples: dict: the dictionary containing all the benign samples
    :param malware_samples: dict: the dictionary containing all the malicious samples
    :param training_ratio: int: How much of the data set should be used for training and by proxy testing
    :param benign_ratio: int: How many benign samples to use in the training data
    :return: A dictionary of samples to use for training data and a dictionary of samples to use as testing data
    """
    training_data = []
    testing_data = []
    for family in malware_families.values():
        sample_count = len(family)
        split_index = int(sample_count * training_ratio)
        training_data.extend(family[:split_index])
        testing_data.extend(family[split_index:])
    print(len(training_data))
    # A lot of setup
    #malware = list(malware_samples.keys())
    #benign = list(benign_samples.keys())
    #malware_count = len(malware)
    #benign_count = int(malware_count * benign_ratio)
    #malware_split_index = int(malware_count * training_ratio)
    #benign_split_index = int(benign_count * training_ratio)
    #benign_data = random.sample(benign, benign_count)

    # just to split the data
    #training_data = malware[:malware_split_index]
    #training_data.extend(benign_data[:benign_split_index])
    #test_data = malware[malware_split_index:]
    #test_data.extend(benign_data[benign_split_index:])

    return dict.fromkeys(training_data), dict.fromkeys(testing_data)


def create_training_and_testing_sets(benign_samples: dict, malware_samples: dict, training_ratio: int,
                                     benign_ratio: int):
    """
    This function builds a training and testing data set from the overall data set. The function under samples the
    benign samples according to the benign ratio.
    :param benign_samples: dict: the dictionary containing all the benign samples
    :param malware_samples: dict: the dictionary containing all the malicious samples
    :param training_ratio: int: How much of the data set should be used for training and by proxy testing
    :param benign_ratio: int: How many benign samples to use in the training data
    :return: A dictionary of samples to use for training data and a dictionary of samples to use as testing data
    """

    # A lot of setup
    malware = list(malware_samples.keys())
    benign = list(benign_samples.keys())
    malware_count = len(malware)
    benign_count = int(malware_count * benign_ratio)
    malware_split_index = int(malware_count * training_ratio)
    benign_split_index = int(benign_count * training_ratio)
    benign_data = random.sample(benign, benign_count)

    # just to split the data
    training_data = malware[:malware_split_index]
    training_data.extend(benign_data[:benign_split_index])
    test_data = malware[malware_split_index:]
    test_data.extend(benign_data[benign_split_index:])

    return dict.fromkeys(training_data), dict.fromkeys(test_data)


def get_features(training_data: dict, benign_samples: dict, malware_samples: dict):
    """
    Creates a feature set from all the training data
    :param training_data: dict: A dictionary of all the training samples
    :param benign_samples: dict: A dictionary of all the benign samples
    :param malware_samples: dict: A dictionary of all the malware samples
    :return: A set of all the features to use for our model
    """
    features = set()

    for sample in training_data.keys():
        if sample in malware_samples.keys():
            features.update(malware_samples[sample])
        else:
            features.update(benign_samples[sample])

    return features


def build_vectors(data: dict, features: set, benign_samples: dict, malware_samples: dict):
    """
    Creates a feature vector for each sample in the dictionary and sets the value to the vector
    :param data: dict: The dictionary of samples to create vectors for
    :param features: set: The set of features to use
    :param benign_samples: dict: The dictionary of all benign samples
    :param malware_samples: dict: The dictionary of all malicious samples
    :return: None
    """

    for sample in data:
        feature_vector = dict.fromkeys(features, 0)
        if sample in malware_samples.keys():
            sample_features = malware_samples[sample]
        else:
            sample_features = benign_samples[sample]

        for feature in sample_features:
            if feature in features:
                feature_vector[feature] = 1

        data[sample] = list(feature_vector.values())


def prepare_data(data_directory: str, label_file: str, training_ratio: int, benign_ratio: int):
    """
    Loads the samples, loads the labels, and creates our vectors
    :param data_directory: str: the zipped directory of samples
    :param label_file:  str: the file of labels
    :return: A unique set of features, a dictionary of samples and their feature vectors, and a dictionary of labels
             and the samples belonging to those labels
    """
    labels, malware_families = load_label_data(label_file)
    benign_samples, malware_samples = load_sample_data(data_directory, labels)
    training_data, testing_data = create_training_and_testing_sets(benign_samples, malware_samples,
                                                                   training_ratio, benign_ratio)
    features = get_features(training_data, benign_samples, malware_samples)
    build_vectors(training_data, features, benign_samples, malware_samples)
    build_vectors(testing_data, features, benign_samples, malware_samples)

    return features, training_data, testing_data, labels, malware_families


def prepare_data_ova(data_directory: str, label_file: str, training_ratio: int, benign_ratio: int):
    """
    Loads the samples, loads the labels, and creates our vectors
    :param data_directory: str: the zipped directory of samples
    :param label_file:  str: the file of labels
    :return: A unique set of features, a dictionary of samples and their feature vectors, and a dictionary of labels
             and the samples belonging to those labels
    """
    labels, malware_families = load_label_data(label_file)
    benign_samples, malware_samples = load_sample_data(data_directory, labels)
    training_data, testing_data = create_training_and_testing_sets_ova(malware_samples,
                                                                   training_ratio, malware_families)
    features = get_features(training_data, benign_samples, malware_samples)
    build_vectors(training_data, features, benign_samples, malware_samples)
    build_vectors(testing_data, features, benign_samples, malware_samples)

    return features, training_data, testing_data, labels, malware_families
