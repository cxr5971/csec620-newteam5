#author: Cullen Rezendes
#file: naive_bayes.py
#purpose: Multinomial Naive Bayes ML on Spam/Ham dataset

from math import *
import tokenizer
import time

#Function: get_message_percentages
#Parameters: labels - this is the list of ham/spam labels, we can use this to identify how many spam and ham messages we have
#Description: Takes the set of ham/spam labels and counts the number of ham and spam messages
#Returns: list of num_spam and num_ham messages
def get_message_percentages(labels):
    
    num_of_spam = 0
    num_of_ham = 0
    for item in labels:
        if item == "spam":
            num_of_spam += 1
        if item == "ham":
            num_of_ham += 1

    perc_spam = float(num_of_spam/len(labels))
    perc_ham = float(num_of_ham/len(labels))
    return [perc_spam, perc_ham]
    

#Function: naive_bayes
#Parameters: training_set - messages to be used in naive bayes training, training_labes - labels that correspond to training set (ham or spam)
#Description: Training naive bayes classifier to detect spam and ham. Applies the Naive Bayes algorithm and gives back the associated parameters for each word
#Returns: list of ham and spam parameters to be used in classification
def naive_bayes(training_set, training_labels):
    #Call the get_message_percentages function to get the percentages of ham and spam
    results = get_message_percentages(training_labels)
    #Get percentage of spam&ham in overall training set
    percentage_spam = results[0]
    percentage_ham = results[1]

    print("Percentage of spam in dataset: " + str(percentage_spam))
    print("Percentage of ham in dataset: " + str(percentage_ham))




    #Count will be our index into the dataset. We will use it to check the corresponding label
    # for each message
    count = 0

    #We want to track the overall number of spam words, so we have a count and list to store words
    num_of_spam_words = 0
    n_words_per_spam = []
    #We do the same for ham
    num_of_ham_words = 0
    n_words_per_ham = []

    #Here we will track how many times we see each word in spam messages and ham messages
    occurence_tracker_spam = {}
    occurence_tracker_ham = {}
    #List of all words that will be seen
    vocab_list = []

    #Go through the training set, count everytime a word is seen (occurence_tracker)
    #  and add it to the vocab_list. Additionally, count how many words are in each spam/ham message 
    while count <= len(training_set)-1:
        if training_labels[count] == "spam":
            n_words_per_spam.append(len(training_set[count])) #Keep track of the word count per smsg
            for word in training_set[count]:
                if word in occurence_tracker_spam:
                    occurence_tracker_spam[word] += 1 #Count how many times a word has been seen
                else:
                    occurence_tracker_spam[word] = 1 #Add word to spam tracker
                num_of_spam_words += 1 #Increment number of spam words, total, repeat words counted
                vocab_list.append(word) #Add the word to the vocab list
        elif training_labels[count] == "ham":
            n_words_per_ham.append(len(training_set[count])) #Keep track of the word count per hmsg
            for word in training_set[count]:
                if word in occurence_tracker_ham:
                    occurence_tracker_ham[word] += 1 #Count how many times a words has been seen
                else:
                    occurence_tracker_ham[word] = 1 #Add word to ham tracker
                vocab_list.append(word) #Add word to the vocab list
                num_of_ham_words += 1 #Increment number of ham words, total, repeat words counted
        count += 1 #Increase the index counter

    #Remove duplicates by using set
    vocab_list = set(vocab_list)
    vocab_len = len(vocab_list)

    #Now we want to calculate the spam and ham parameters for each word
    spam_param = {}
    ham_param = {}
    print("Num of spam words: " + str(num_of_spam_words))
    print("Num of ham words: " + str(num_of_ham_words))
    
    #For word in vocab list, if the word is seen at all in spam, we calculate it's occurence
    # divided by the number of spam words plus vocab length. We add 1 to the numerator and 
    # vocab length to avoid the zero frequency problem
    for word in vocab_list:
        if word in occurence_tracker_spam:
            n_w_given_spam = occurence_tracker_spam[word]
            p_w_given_spam = (occurence_tracker_spam[word]+1 / (num_of_spam_words+(1*vocab_len)))
            spam_param[word] = p_w_given_spam
        if word in occurence_tracker_ham:
            n_w_given_ham = occurence_tracker_ham[word]
            p_w_given_ham = (occurence_tracker_ham[word]+1 / (num_of_ham_words+(1*vocab_len)))
            ham_param[word] = p_w_given_ham

    parameters = [ham_param,spam_param]
    return parameters

#Function: classify_test_set
#Parameters: message - the msg to classify, parameter_ham/parameter_spam - the calculated naive bayes values for each word that is either spam or ham (gotten from naive bayes function), percentage_spam/ham - the percentage of ham/spam messages in the dataset
#Description: Takes the calculated parameters from naive bayes function and uses it to classify new messages, applying the likelihoods to each word in a message and multiplying by the overall spam/ham percentage
#Returns 
def classify_test_set(message,parameter_ham,parameter_spam,percentage_spam,percentage_ham):

    #For each word in the message, multiple it by the spam/ham percentage. When done,
    # compare which likelihood is higher, spam or ham
    for word in message:
        if word in parameter_spam:
            percentage_spam *= parameter_spam[word]
        #Thought this else may help when a word hasn't been seen, it improves recall but 
        # hits precision and accuracy 
        #else:
        #    parameter_spam[word] = 0
        if word in parameter_ham:
            percentage_ham *= parameter_ham[word]
        #else:
        #   parameter_ham[word] = 0

    #Here is where we send back our predicted label
    if percentage_ham > percentage_spam:
        return [parameter_ham,parameter_spam,'ham']
    elif percentage_spam > percentage_ham:
        return [parameter_ham,parameter_spam,'spam']
    else:
        return [parameter_ham,parameter_spam,'idk']
	    
#Run the naive_bayes classifier on the training and test set. Calculate the number of TP, TN, FP, FN, as well as accuracy, recall, precision, and F1 score
def main():
    start = time.time()
    dataset = tokenizer.tokenize_bayes()
    #dataset[0] is training set, dataset[1] is training labels, dataset[2] is test set,
    # dataset[3] is test labels
    # Returned parameters are from our training set
    parameters = naive_bayes(dataset[0], dataset[1])

    results = get_message_percentages(dataset[1])
    perc_spam = results[0]
    perc_ham = results[1]
    count = 0
    num_of_correct_messages = 0
    num_of_incorrect_messages = 0
    #We have parameters for our True Positive, False Negative, False Positive, and True Negative
    TP = 0
    FN = 0
    FP = 0
    TN = 0
    param_ham = parameters[0]
    param_spam = parameters[1]
    #Run our machine learning Naive Bayes on the test set using our learned parameters
    for message in dataset[2]:
        results = classify_test_set(message,param_ham,param_spam,perc_spam,perc_ham)
        #I added this parameter returning in case we wanted to modify parameters midway through
        # training but we don't need to
        param_ham = results[0]
        param_spam = results[1]
        predicted_label = results[2]
        if predicted_label == dataset[3][count]:
            num_of_correct_messages += 1
            if predicted_label == 'ham':
                TN += 1
            if predicted_label == 'spam':
                TP += 1
        else:
            num_of_incorrect_messages += 1
            if predicted_label == 'ham':
                FN += 1
            if predicted_label == 'spam':
                FP += 1

        count += 1

    #Perform our result calculations
    accuracy = num_of_correct_messages/len(dataset[2])
    print("TP: " + str(TP))
    print("FP: " + str(FP))
    print("TN: " + str(TN))
    print("FN: " + str(FN))
    print("accuracy: " + str(accuracy))
    precision = (TP/(TP+FP))
    recall = (TP/(TP+FN))
    print("precision: " + str(TP/(TP+FP)))
    print("recall: " + str(TP/(TP+FN)))
    print("F1-score: " + str((2*precision*recall)/(precision+recall)))
    print("correct: " + str(num_of_correct_messages))
    print("incorect: " + str(len(dataset[2])-num_of_correct_messages))
    end = time.time()
    print("Execution Time (seconds): " + str(end - start))




if __name__ == '__main__':
    main()
