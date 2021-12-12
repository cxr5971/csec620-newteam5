

import numpy as np
import pandas as pd
import tables
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from numpy import mean, std
from sklearn import model_selection, feature_selection, linear_model, metrics, svm
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier


X1 = pd.read_hdf('capture20110810.h5', key='data')
X1.reset_index(drop=True, inplace=True)

#X2 = pd.read_hdf('2013-10-01_capture-win12.h5', key='data')
#X2.reset_index(drop=True, inplace=True)

#X2 = pd.read_hdf('data_window3.h5', key='data')
#X2.reset_index(drop=True, inplace=True)

#X = X.join(X2)

X1.drop('window_id', axis=1, inplace=True)
#X2.drop('window_id', axis=1, inplace=True)
y1 = X1['Label_<lambda>']
#y2 = X2['Label_<lambda>']
X1.drop('Label_<lambda>', axis=1, inplace=True)
#X2.drop('Label_<lambda>', axis=1, inplace=True)

labels1 = np.load("capture20110810_labels_12.npy", allow_pickle=True)
#labels2 = np.load("2013-10-01_capture-win12_labels_12.npy", allow_pickle=True)
#print(X)
#print(y)
print(X1.columns.values)
print(labels1)
#print(X2.columns.values)
#print(labels2)


#If y == 6, then y_bin6 is True, else False
y_1 = y1==6
#y_2 = y2==1
#Create training and testing data
X_train, X_test, y_train, y_test = model_selection.train_test_split(X1, y_1, test_size=0.33, random_state=123456)



print("y1", np.unique(y1, return_counts=True))
#print("y_train", np.unique(y_train, return_counts=True))
#print("y_test", np.unique(y_test, return_counts=True))


#Create Random Forest Classifier 
#https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train,y_train)
y_pred=clf.predict(X_test)
print("Random Forest Accuracy")
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred))
print("Recall:",metrics.recall_score(y_test, y_pred))
print("F1-Score:",metrics.f1_score(y_test, y_pred))
tn, fp, fn, tp = metrics.confusion_matrix(y_test,y_pred).ravel()
print("True Negative: " + str(tn))
print("False Positive: " + str(fp))
print("False Negative: " + str(fn))
print("True Positive: " + str(tp))



#cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=10)
#n_scores = cross_val_score(model, X_train, y_train, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
#print('Accuracy: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))

print("\n\n")
print("SVM Accuracy")
#Create SVM Classifier
clf = svm.SVC(kernel='linear') # Linear Kernel
#Train the model using the training sets
clf.fit(X_train, y_train)
#clf.fit(X1,y_1)
#Predict the response for test dataset
y_pred = clf.predict(X_test)
#y_pred = clf.predict(X2)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred))
print("Recall:",metrics.recall_score(y_test, y_pred))
print("F1-Score:",metrics.f1_score(y_test, y_pred))
tn, fp, fn, tp = metrics.confusion_matrix(y_test,y_pred).ravel()
#tn, fp, fn, tp = metrics.confusion_matrix(y_test,y_pred)
print("True Negative: " + str(tn))
print("False Positive: " + str(fp))
print("False Negative: " + str(fn))
print("True Positive: " + str(tp))


print("Logistic Regression")
clf = LogisticRegression(random_state=0).fit(X_train, y_train)
clf.predict(X_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred))
print("Recall:",metrics.recall_score(y_test, y_pred))
print("F1-Score:",metrics.f1_score(y_test, y_pred))
tn, fp, fn, tp = metrics.confusion_matrix(y_test,y_pred).ravel()
#tn, fp, fn, tp = metrics.confusion_matrix(y_test,y_pred)
print("True Negative: " + str(tn))
print("False Positive: " + str(fp))
print("False Negative: " + str(fn))
print("True Positive: " + str(tp))


print("Neural Network")
clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
clf.fit(X_train, y_train)
clf.predict(X_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred))
print("Recall:",metrics.recall_score(y_test, y_pred))
print("F1-Score:",metrics.f1_score(y_test, y_pred))
tn, fp, fn, tp = metrics.confusion_matrix(y_test,y_pred).ravel()
print("True Negative: " + str(tn))
print("False Positive: " + str(fp))
print("False Negative: " + str(fn))
print("True Positive: " + str(tp))
