Introduction:

     This project uses the K-nn and naive bayes classifiers to predict whether SMS messages are spam. The project is
     divided into 3 main files (tokenizer.py, knn.py, and naive_bays.py) plus a corpus of labeled SMS data
     (SMSSpamCollection).

Usage:

     This project does use the NLTK library while processing the SMS data. Specifically we use the NLTK library to
     remove stop words and stem our tokens. If you do not have the NLTK library already you will need to install it. To
     install the library simply open a terminal and run:

              pip3 install nltk

     Once this is finished the library should be installed, however it may require further updating. To update the NLTK
     library run the following command in a terminal:

              python -m nltk.downloader all

     Now the NLTK library should be fully installed and updated.

     To run the k-nn classifier simply run:

              python knn.py

     This will run the knn classifier twice with k = 5. One run uses the Euclidean distance for it distance metric while
     the other uses the cosine similarity to find the nearest neighbors. This takes a most 5 minutes to run. There are
     some progress messages to help show the classifier is still running and how much longer it will take. If you want
     to run the classifier with a different value for K, simply open knn.py and edit the k variable on line 339 of the
     file. Similarly if you want to only run the classifier with one metric, you can comment out the undesired
     classifier in the main function. The call to the euclidean classifier occurs on line 344 while the cosine
     classifier is called on line 350

     To run the naive bayes classifier run:

              python naive_bayes.py

Files:

     tokenizer.py:

          This file contains all the logic for processing the SMS messages into usable data for each classifier. The
          file reads in the data from a specified data file and tokenizes each message. In this tokenization process the
          messages are split on whitespace, lowercased, punctuation and stop words are stripped, and the tokens are
          stemmed using a PorterStemmer implemented in the NLTK python library

     knn.py:

          This file uses the K-NN classifier to predict whether an SMS message is spam. The features used for
          classification are all the unique words that appear in the training data. For each message a feature vector
          is created by calculating the TF-IDF value of each word in the message. The classifier then computes the
          distance between our unknown vector and all other feature vectors in the training data. To calculate this
          distance the file uses both the Euclidean distance and cosine similarity between the two points. Once the
          distances are calculated, the classifier predicts whether or not the given message is spam. This file also
          contains logic for reporting the results of the classifier.

     naive_bayes.py:

          This file uses the Naive Bayes classifier to predict whether an SMS message is spam. The features used for classification are the frequency of seeing words in the training data. For each message, the frequency of seeing a word in a spam or ham message is tracked. Additionally, all words are added to a vocab word list and the number of how many words are tracked. The Naive Bayes Theorem is used to calculate the probability of a message being spam ( P(Spam | W) ) and the probability of a message being ham ( P(Ham | W)). As an example, the probability of a message being spam is the probability of spam messages overall times all of probabilities (multiplied) of each word with the assumption that the message is spam. Once all the calculations are complete, the P(Spam|W) is compared to P(Ham|W), and whichever is higher is taken as the classification. Another note is the zero frequency problem. This occurs when a word has never been seen in a spam or ham message. This causes a zero probability which will result in the entire probability of spam or ham being zero. To account for this, we assume that if a word has not been seen, it instead is replaced with one. Every other value is increased by one as well, so that the overall scale does not change. This will cancel out any word that has not been seen, as any value times one is just that same value. 
