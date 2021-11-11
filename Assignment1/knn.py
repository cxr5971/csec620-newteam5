import math
from tokenizer import tokenize_knn


def k_nearest_neighbors_euclidean(k, training_samples, training_labels, testing_samples):
    """
    The method that actually runs the K-NN algorithm. This method handles all the TF-IDF calculations, distance
    calculations using euclidean distance, and predictions
    :param k: int: The number of nearest neighbors to look at for our predictions
    :param training_samples: dict(str, list[str]): A dictionary mapping a given training message to its tokenized
        equivalent
    :param training_labels: dict(str, str): A dictionary mapping a given training message to its label
    :param testing_samples: dict(str, list[str]): A dictionary mapping a test message to its tokenized form
    :return: A dictionary mapping the test messages to their predicted spam/ham categories
    """

    # Just some simple code to allow for progress messages
    # More helpful than just staring at a blank screen imo
    samples_to_predict = len(testing_samples.keys())
    samples_predicted = 0
    ten_percent = samples_to_predict // 10
    current_percentage = 0

    # Step 1, create a set of unique words
    unique_words = set()
    for sample in training_samples.values():
        # simply iterate over the training set and condense to a set
        unique_words.update(sample)

    # Step 2, calculate the IDF
    idf_dict = dict.fromkeys(unique_words, 0.0)  # Make a dict of all unique words
    """
    This loop takes each SMS sample, condenses it into a set of unique words, and then updates the value in the
    idf dictionary by 1. This servers as a count of how many samples each word occurred in.
    """
    for sample in training_samples.values():
        temp_set = set()
        temp_set.update(sample)
        for word in temp_set:
            idf_dict[word] = idf_dict[word] + 1

    """
    Once we have read through all the samples, the dictionary should hold how many samples each word occurred in. Now
    we simply loop over the dictionary and update each value to be the actual Inverse Document Frequency. For this 
    we do log10(len(training_set) / idf_dict[key]) since the length of the training set should be the total number of
    samples and idf_dict[key] should be the number of samples each word appeared in
    """
    total_samples = len(training_samples.values())
    for key in idf_dict.keys():
        idf_dict[key] = math.log10(total_samples / idf_dict[key])

    # Step 3, calculate TF + Step 4, make feature vectors
    training_tfidf_dict = calculate_tfidf(training_samples, idf_dict)
    test_tfidf_dict = calculate_tfidf(testing_samples, idf_dict)

    #  Classify/predict
    predicted_dict = dict()

    for message, message_vector in test_tfidf_dict.items():
        distances = []
        for training_message, training_vector in training_tfidf_dict.items():
            # Simple base case for empty vectors i.e. completely new sample
            # Though honestly Im not 100% sure where some are coming from
            if message_vector == {} or training_vector == {}:
                break

            """
            To speed up calculations, rather than have a vector of len(unique_words), each vector only contains the 
            words in that sample. To do this we combine both vectors words into one key set. Then if a word in the key 
            set doesn't appear in a vector, we simply use a 0 for the TF-IDF value. This mimics having the complete 
            vector without having to loop over thousands of words for thousands of samples. TLDR time savings
            """
            key_set = set()
            key_set.update(message_vector.keys())
            key_set.update(training_vector.keys())
            distance = calculate_euclidean_distance(key_set, message_vector, training_vector)
            distances.append((training_message, distance))

        # Once all distances have been calculated, sort and choose the K nearest neighbors
        distances.sort(key=lambda x: x[1])
        nearest_neighbors = distances[:k]

        # Now we simply count our results
        spam = ham = 0
        for neighbor in nearest_neighbors:
            if training_labels[neighbor[0]] == "spam":
                spam += 1
            elif training_labels[neighbor[0]] == "ham":
                ham += 1

        # And based on that we predict
        if spam > ham:
            predicted_dict[message] = "spam"
        elif ham > spam:
            predicted_dict[message] = "ham"
        else:
            predicted_dict[message] = "error"  # There is the possibility of errors occurring

        # Here is the pay off for our percentage remaining info messages
        samples_predicted += 1
        if samples_predicted % ten_percent == 0:
            current_percentage += 10
            print(f"{current_percentage}% of samples processed")

    # This is where we would return once things are working
    return predicted_dict


def k_nearest_neighbors_cosine(k, training_samples, training_labels, testing_samples):
    """
    The method that actually runs the K-NN algorithm. This method handles all the TF-IDF calculations, distance
    calculations using cosine similarity, and predictions
    :param k: int: The number of nearest neighbors to look at for our predictions
    :param training_samples: dict(str, list[str]): A dictionary mapping a given training message to its tokenized
        equivalent
    :param training_labels: dict(str, str): A dictionary mapping a given training message to its label
    :param testing_samples: dict(str, list[str]): A dictionary mapping a test message to its tokenized form
    :return: A dictionary mapping the test messages to their predicted spam/ham categories
    """

    # Just some simple code to allow for progress messages
    # More helpful than just staring at a blank screen imo
    samples_to_predict = len(testing_samples.keys())
    samples_predicted = 0
    ten_percent = samples_to_predict // 10
    current_percentage = 0

    # Step 1, create a set of unique words
    unique_words = set()
    for sample in training_samples.values():
        # simply iterate over the training set and condense to a set
        unique_words.update(sample)

    # Step 2, calculate the IDF
    idf_dict = dict.fromkeys(unique_words, 0.0)  # Make a dict of all unique words
    """
    This loop takes each SMS sample, condenses it into a set of unique words, and then updates the value in the
    idf dictionary by 1. This servers as a count of how many samples each word occurred in.
    """
    for sample in training_samples.values():
        temp_set = set()
        temp_set.update(sample)
        for word in temp_set:
            idf_dict[word] = idf_dict[word] + 1

    """
    Once we have read through all the samples, the dictionary should hold how many samples each word occurred in. Now
    we simply loop over the dictionary and update each value to be the actual Inverse Document Frequency. For this 
    we do log10(len(training_set) / idf_dict[key]) since the length of the training set should be the total number of
    samples and idf_dict[key] should be the number of samples each word appeared in
    """
    total_samples = len(training_samples.values())
    for key in idf_dict.keys():
        idf_dict[key] = math.log10(total_samples / idf_dict[key])

    # Step 3, calculate TF + Step 4, make feature vectors
    training_tfidf_dict = calculate_tfidf(training_samples, idf_dict)
    test_tfidf_dict = calculate_tfidf(testing_samples, idf_dict)

    #  Classify/predict
    predicted_dict = dict()

    for message, message_vector in test_tfidf_dict.items():
        distances = []
        for training_message, training_vector in training_tfidf_dict.items():
            # Simple base case for empty vectors i.e. completely new sample
            # Though honestly Im not 100% sure where some are coming from
            if message_vector == {} or training_vector == {}:
                break

            """
            To speed up calculations, rather than have a vector of len(unique_words), each vector only contains the 
            words in that sample. To do this we combine both vectors words into one key set. Then if a word in the key 
            set doesn't appear in a vector, we simply use a 0 for the TF-IDF value. This mimics having the complete 
            vector without having to loop over thousands of words for thousands of samples. TLDR time savings
            """
            key_set = set()
            key_set.update(message_vector.keys())
            key_set.update(training_vector.keys())
            distance = calculate_cosine_similarity(key_set, message_vector, training_vector)
            distances.append((training_message, distance))

        # Once all distances have been calculated, sort and choose the K nearest neighbors
        # Reversed because its cosine similarity
        distances.sort(key=lambda x: x[1], reverse=True)
        nearest_neighbors = distances[:k]

        # Now we simply count our results
        spam = ham = 0
        for neighbor in nearest_neighbors:
            if training_labels[neighbor[0]] == "spam":
                spam += 1
            elif training_labels[neighbor[0]] == "ham":
                ham += 1

        # And based on that we predict
        if spam > ham:
            predicted_dict[message] = "spam"
        elif ham > spam:
            predicted_dict[message] = "ham"
        else:
            predicted_dict[message] = "error"  # There is the possibility of errors occurring

        # Here is the pay off for our percentage remaining info messages
        samples_predicted += 1
        if samples_predicted % ten_percent == 0:
            current_percentage += 10
            print(f"{current_percentage}% of samples processed")

    # This is where we would return once things are working
    return predicted_dict


def calculate_tfidf(sample_dict, idf_dict):
    """
    This loop is a little complicated. First we loop over all the samples in the sample_set. We condense each sample
    into a set of unique words for each sample. Then we call a helper function to get the word count of each message.
    Next we loop over these words and calculate the TF-IDF value if we have an IDF value for it. We store these values
    in a dictionary of word, TF-IDF key value pairs to represent our feature vectors. To speed up calculations later in
    the program, we only store words from the sample in this dictionary. Then we store each vector in a dictionary with
    the message as its key.
    :param sample_dict: dict(str, list[str]): A dictionary mapping a message to its tokenized form
    :param idf_dict: dict(str, int): a lookup table for the IDF values of each of our features
    :return: A dictionary mapping the message to a dictionary that contains the TF-IDF value for each word in the
        tokenized message (if said word is a feature)
    """
    tfidf_dict = dict()
    for message, tokens in sample_dict.items():
        sample_dict = dict()
        temp_set = set()
        word_count = process_message(message)
        temp_set.update(tokens)
        for word in temp_set:
            # token.count counts the number of occurrences of each word in the sample
            if word in idf_dict:  # Only calculate if it appears in out IDF values
                sample_dict[word] = (tokens.count(word) / word_count) * idf_dict[word]

        tfidf_dict[message] = sample_dict
    return tfidf_dict


def process_message(message):
    """
    A method that simply counts the number of word in the message. The reason for doing this rather than splitting and
    using the len method was to screen out any empty strings from throwing off the count. Yes this was an issue in the
    past
    :param message: str: The message to be counted
    :return: The number of words in the message
    """
    word_count = 0
    for word in message.split(" "):
        if word.strip() != "":
            word_count += 1
    return word_count


def calculate_euclidean_distance(key_set, message_vector, training_vector):
    """
    A simple Euclidean distance calculation for x dimensions. As I mentioned to speed up calculations each feature
    vector on contains features with a non-zero value. To address this we combine both vectors' keys into a key set and
    if a key is missing from either vector (i.e. the word doesn't appear in the sample) we simply use a 0 in the
    calculation. This cuts the loop from the entire feature set to just tens of words per calculation.
    :param key_set: set(str): The union of features in both the test and training vector
    :param message_vector: dict(str, int): A dictionary representing the feature vector of our test message. THe feature
        vector composes of the TF-IDF values calculated earlier
    :param training_vector: dict(str, int): Another dictionary representation of the feature vector, but now for the
        training message
    :return: The euclidean distance between the two messages
    """

    running_total = 0
    for key in key_set:
        test_point = message_vector.get(key, 0)
        training_point = training_vector.get(key, 0)
        running_total += (test_point - training_point)**2
    distance = math.sqrt(running_total)
    return distance


def calculate_cosine_similarity(key_set, message_vector, training_vector):
    """
    This method calculates the cosine similarity of our test message and some other message in our training set
    :param key_set: set(str): Again this is the union of features between both messages
    :param message_vector: dict(str, int): A dictionary representing the feature vector for our test message
    :param training_vector: dict(str, int): Still a dictionary representing a feature vector but this time for the
        training message
    :return: The cosine similarity of the two messages
    """
    message_square = 0
    for value in message_vector.values():
        message_square += value**2
    message_square = math.sqrt(message_square)

    sample_square = 0
    for value in training_vector.values():
        sample_square += value**2
    sample_square = math.sqrt(sample_square)

    dot_product = 0
    for key in key_set:
        dot_product += message_vector.get(key, 0) * training_vector.get(key, 0)
    return dot_product / (message_square * sample_square)


def compute_results(predictions, actual):
    """
    This function does just as it is named, it calculates all of our statistics for us.
    :param predictions: dict(str, str): A dictionary mapping each test message to our predicted value
    :param actual: dict(str, str): A dictionary mapping each test message to its actual label
    :return: Nothing
    """
    true_positives = false_positives = true_negatives = false_negatives = 0

    for key in predictions.keys():
        prediction = predictions[key]
        if prediction == "spam":
            if actual[key] == "spam":
                true_positives += 1
            elif actual[key] == "ham":
                false_positives += 1
        elif prediction == "ham":
            if actual[key] == "ham":
                true_negatives += 1
            elif actual[key] == "spam":
                false_negatives += 1

    accuracy = (true_positives + true_negatives) / (true_positives + true_negatives + false_positives + false_negatives)
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    f_one = (2 * precision * recall) / (precision + recall)

    print(f"\tTrue Positives: {true_positives}\n\tTrue Negatives: {true_negatives}"
          f"\n\tFalse Positives: {false_positives}\n\tFalse Negatives: {false_negatives}\n\tAccuracy: {accuracy}"
          f"\n\tPrecision: {precision}\n\tRecall: {recall}\n\tF1-score: {f_one}")


def main():

    k = 5
    print("Loading the data set from SMSSpamCollection")
    training_samples, training_labels, testing_samples, testing_labels = tokenize_knn()

    print(f"Computing K-NN with a value of k = {k} and using Euclidean distance as the distance metric")
    predictions = k_nearest_neighbors_euclidean(k, training_samples, training_labels, testing_samples)

    print(f"Calculating the results with k = {k} using Euclidean distance:")
    compute_results(predictions, testing_labels)

    print(f"Computing K-NN with a value of k = {k} and using cosine similarity as the distance metric")
    predictions = k_nearest_neighbors_cosine(k, training_samples, training_labels, testing_samples)

    print(f"Calculating the results with k = {k} using cosine similarity:")
    compute_results(predictions, testing_labels)


if __name__ == '__main__':
    main()

