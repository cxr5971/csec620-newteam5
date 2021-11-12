import string

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

FILENAME = 'SMSSpamCollection'
TRAINING_RATIO = .7
TESTING_RATIO = 1 - TRAINING_RATIO

def load_dataset(fp):
    """
    A function that loads in the data set and tokenizes each message
    :param fp: A string representing the name of the file that contains the SMS data
    :return: A list of all the labels, labels, and a list containing the tokenized SMS messages, tokenized_messages
    """
    labels = list()
    messages = list()

    with open(fp, "r") as infile:
        for line in infile:
            line = line.strip()
            temp = line.split("\t", 1)  # Turns out the labels and messages are separated by a \t
            labels.append(temp[0])
            messages.append(temp[1])

    return labels, messages


def tokenize_messages(messages):
    """
    This function splits a message into individual tokens. This method creates necessary variables for the tokenization
    process before passing them off to a helper function for the actual cleaning of input
    :param messages: str: A message to be converted into tokens
    :return: A list of cleaned tokens that represent the provided message
    """
    port_stem = PorterStemmer()
    punctuation_table = str.maketrans("", "", string.punctuation)
    stopword_set = set(stopwords.words("english"))
    tokenized_messages = list()

    for message in messages:
        tokenized_messages.append(tokenize_messages_helper(message, punctuation_table, stopword_set, port_stem))

    return tokenized_messages


def tokenize_messages_helper(message, punctuation_table, stopword_set, porter_stemmer):
    """
    This function splits each message into individual words and then cleans the data
    :param message: A string representing an SMS message
    :param punctuation_table: A mapping table of punctuation marks to remove from our tokens
    :param stopword_set: A set of english stopwords to remove from our tokens
    :param porter_stemmer: A NLTK PorterStemmer to stem our tokens
    :return: A list of the cleaned words from the text message
    """
    cleaned_tokens = list()
    raw_tokens = message.split(" ")

    # This loop converts each word to lowercase and strips any punctuation from the word
    for word in raw_tokens:
        word = word.lower()
        word = word.translate(punctuation_table)

        # If allowed this uses NLTK to remove stopwords and stem the words
        if word not in stopword_set:
            word = porter_stemmer.stem(word)
            cleaned_tokens.append(word)

    return cleaned_tokens


def tokenize_bayes():
    """
    A function to load the data set and process the messages for the naive bayes classifier. This tokenization process
    does not require the original message to be returned along with the messages tokens
    :return:
    """
    labels, messages = load_dataset(FILENAME)
    tokens = tokenize_messages(messages)

    split_index = int(len(labels) * TRAINING_RATIO)
    training_set = tokens[:split_index]
    training_labels = labels[:split_index]
    testing_set = tokens[split_index:]
    testing_labels = labels[split_index:]
    return [training_set, training_labels, testing_set, testing_labels]


def tokenize_knn():
    """
    A function to load the dataset from our file and process it for use with the K-NN algorithm. This involves splitting
    the data into test and training sets as well as tokenizing the data. K-NN also requires we return the original
    messages as well for the term frequency calculations
    :return: A series of dictionaries mapping messages to their coresponding tokens and labels for both the training and
        testing data
    """
    labels, messages = load_dataset(FILENAME)
    tokens = tokenize_messages(messages)

    split_index = int(len(labels) * TRAINING_RATIO)
    training_messages = messages[:split_index]
    training_set = tokens[:split_index]
    training_labels = labels[:split_index]

    testing_messages = messages[split_index:]
    testing_set = tokens[split_index:]
    testing_labels = labels[split_index:]

    return dict(zip(training_messages, training_set)), dict(zip(training_messages, training_labels)), \
        dict(zip(testing_messages, testing_set)), dict(zip(testing_messages, testing_labels))
