

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
X_mal = X1.loc[X1['Label_<lambda>'] == 6]
mal_count = X_mal.shape[0]
print(mal_count)
benign_count = int(mal_count*10)
X_benign = X1[X1['Label_<lambda>'] != 6]
X_combined = pd.concat([X_mal, X_benign.sample(n=benign_count, random_state=123456) ])

X_combined.drop('window_id', axis=1, inplace=True)
#X1.drop('window_id', axis=1, inplace=True)
#X2.drop('window_id', axis=1, inplace=True)
y1 = X_combined['Label_<lambda>']
#y1 = X1['Label_<lambda>']
#y2 = X2['Label_<lambda>']
#X1.drop('Label_<lambda>', axis=1, inplace=True)
#X2.drop('Label_<lambda>', axis=1, inplace=True)
X_combined.drop('Label_<lambda>', axis=1, inplace=True)
#X2.drop('Label_<lambda>', axis=1, inplace=True)

labels1 = np.load("capture20110810_labels_12.npy", allow_pickle=True)
#labels2 = np.load("2013-10-01_capture-win12_labels_12.npy", allow_pickle=True)
#print(X)
#print(y)
print(X_combined.columns.values)
print(labels1)
#print(X2.columns.values)
#print(labels2)


#If y == 6, then y_bin6 is True, else False
y_1 = y1==6
#y_2 = y2==1
#Create training and testing data
X_train, X_test, y_train, y_test = model_selection.train_test_split(X_combined, y_1, test_size=0.33, random_state=123456)



print("y1", np.unique(y1, return_counts=True))
#print("y_train", np.unique(y_train, return_counts=True))
#print("y_test", np.unique(y_test, return_counts=True))


#Create Random Forest Classifier 
#https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
with open('rf_1_10_n_iterations_50_400','w+') as f_name:
    f_name.write('Random Forest Min Sample Split 2 - 10\n')
    f_name.write('Accuracy,Precision,Recall,F1-Score,AUC,True_Negative,False_Positive,False_Negative,True_Positive\n')
    for i in range(50,400,50):
        f_name.write(str(i))
        print("Random Forest Classifier")
        clf = RandomForestClassifier(n_estimators=i)
        clf.fit(X_train,y_train)
        y_pred=clf.predict(X_test)
	#print("Accuracy: " +str(metrics.accuracy_score(y_test, y_pred)))
	#print("Precision: " +str(metrics.precision_score(y_test, y_pred)))
	#print("Recall: " +str(metrics.recall_score(y_test, y_pred)))
	#print("F1-Score: " +str(metrics.f1_score(y_test, y_pred)))
        data = str(metrics.accuracy_score(y_test, y_pred)) + ','
        data += str(metrics.precision_score(y_test, y_pred)) + ','
        data += str(metrics.recall_score(y_test, y_pred)) + ','
        data += str(metrics.f1_score(y_test, y_pred)) + ','
        tn, fp, fn, tp = metrics.confusion_matrix(y_test,y_pred).ravel()
        fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred)
        data += str(metrics.auc(fpr, tpr)) + ','
#print("True Negative: " + str(tn))
#print("False Positive: " + str(fp))
#print("False Negative: " + str(fn))
#print("True Positive: " + str(tp))
        data += str(tn) + ','
        data += str(fp) + ','
        data += str(fn) + ','
        data += str(tp)
        f_name.write(data+'\n')

#importances = clf.feature_importances_
#forest_importances = pd.Series(importances, index=X_combined.columns.values)
#std = np.std([tree.feature_importances_ for tree in clf.estimators_], axis=0)
#fig, ax = plt.subplots()
#forest_importances.plot.bar(yerr=std, ax=ax)
#ax.set_title("Feature importances using MDI")
#ax.set_ylabel("Mean decrease in impurity")
#fig.tight_layout()
#plt.show()
#cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=10)
#n_scores = cross_val_score(model, X_train, y_train, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
#print('Accuracy: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))

print("\n")

#svc_args={'loss':'hinge', 'penalty':'elasticnet', 'max_iter':1000, 'alpha':1e-9, 'tol':1e-3, 'random_state':123456, 'class_weight':None}
#kernel_args={'kernel':'rbf', 'gamma':None, 'degree':None, 'n_components':100, 'random_state':123456}
#clf = linear_model.SGDClassifier(**svc_args)
#cv = model_selection.ShuffleSplit(n_splits=10, test_size=0.1, random_state=123456)
#scores = model_selection.cross_validate(clf, X_train, y_train, cv=cv, scoring=['accuracy','precision', 'recall', 'f1'], return_train_score=True)
#print(scores)
#print(str(np.mean(scores['test_accuracy'])))
#print(str(np.mean(scores['test_precision'])))
#print(str(np.mean(scores['test_recall'])))
#print(str(np.mean(scores['test_f1'])))



print("SVM Accuracy")
    #Create SVM Classifier
clf = svm.SVC(gamma='auto',max_iter=1000,kernel='rbf') # Linear Kernel
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

with open('svm_1_10_max_iter_1000_5000','w+') as f_name:
    f_name.write('\n')
    f_name.write('Accuracy,Precision,Recall,F1-Score,AUC,True_Negative,False_Positive,False_Negative,True_Positive\n')
    for i in range(1000,5000,500):
        f_name.write(str(i))
        print("SVM")
        clf = svm.SVC(max_iter=i,kernel='rbf') # Linear Kernel
        #Train the model using the training sets
        clf.fit(X_train, y_train)
        #clf.fit(X1,y_1)
        #Predict the response for test dataset
        y_pred = clf.predict(X_test)
	#print("Accuracy: " +str(metrics.accuracy_score(y_test, y_pred)))
	#print("Precision: " +str(metrics.precision_score(y_test, y_pred)))
	#print("Recall: " +str(metrics.recall_score(y_test, y_pred)))
	#print("F1-Score: " +str(metrics.f1_score(y_test, y_pred)))
        data = str(metrics.accuracy_score(y_test, y_pred)) + ','
        data += str(metrics.precision_score(y_test, y_pred)) + ','
        data += str(metrics.recall_score(y_test, y_pred)) + ','
        data += str(metrics.f1_score(y_test, y_pred)) + ','
        fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred)
        data += str(metrics.auc(fpr, tpr)) + ','
        tn, fp, fn, tp = metrics.confusion_matrix(y_test,y_pred).ravel()
#print("True Negative: " + str(tn))
#print("False Positive: " + str(fp))
#print("False Negative: " + str(fn))
#print("True Positive: " + str(tp))
        data += str(tn) + ','
        data += str(fp) + ','
        data += str(fn) + ','
        data += str(tp)
        f_name.write(data+'\n')

print("\n")
print("Logistic Regression")
clf = linear_model.LogisticRegression(random_state=0).fit(X_train, y_train)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
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
with open('logr_1_10_c_1_30','w+') as f_name:
    f_name.write('\n')
    f_name.write('Accuracy,Precision,Recall,F1-Score,AUC,True_Negative,False_Positive,False_Negative,True_Positive\n')
    for i in range(1,30):
        f_name.write(str(i))
        print("LogR")
	#print("Accuracy: " +str(metrics.accuracy_score(y_test, y_pred)))
	#print("Precision: " +str(metrics.precision_score(y_test, y_pred)))
	#print("Recall: " +str(metrics.recall_score(y_test, y_pred)))
	#print("F1-Score: " +str(metrics.f1_score(y_test, y_pred)))
        clf = linear_model.LogisticRegression(C=i,random_state=0).fit(X_train, y_train)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        data = str(metrics.accuracy_score(y_test, y_pred)) + ','
        data += str(metrics.precision_score(y_test, y_pred)) + ','
        data += str(metrics.recall_score(y_test, y_pred)) + ','
        data += str(metrics.f1_score(y_test, y_pred)) + ','
        fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred)
        data += str(metrics.auc(fpr, tpr)) + ','
        tn, fp, fn, tp = metrics.confusion_matrix(y_test,y_pred).ravel()
#print("True Negative: " + str(tn))
#print("False Positive: " + str(fp))
#print("False Negative: " + str(fn))
#print("True Positive: " + str(tp))
        data += str(tn) + ','
        data += str(fp) + ','
        data += str(fn) + ','
        data += str(tp)
        f_name.write(data+'\n')



print("\n")
print("Neural Network")
clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred))
print("Recall:",metrics.recall_score(y_test, y_pred))
print("F1-Score:",metrics.f1_score(y_test, y_pred))
tn, fp, fn, tp = metrics.confusion_matrix(y_test,y_pred).ravel()
print("True Negative: " + str(tn))
print("False Positive: " + str(fp))
print("False Negative: " + str(fn))
print("True Positive: " + str(tp))

with open('nn_1_10_max_iter_200_1000','w+') as f_name:
    f_name.write('\n')
    f_name.write('Accuracy,Precision,Recall,F1-Score,AUC,True_Negative,False_Positive,False_Negative,True_Positive\n')
    for i in range(200,1000):
        f_name.write(str(i))
        print("NeuralNetwork")
	#print("Accuracy: " +str(metrics.accuracy_score(y_test, y_pred)))
	#print("Precision: " +str(metrics.precision_score(y_test, y_pred)))
	#print("Recall: " +str(metrics.recall_score(y_test, y_pred)))
	#print("F1-Score: " +str(metrics.f1_score(y_test, y_pred)))
        clf = MLPClassifier(max_iter=i, solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        data = str(metrics.accuracy_score(y_test, y_pred)) + ','
        data += str(metrics.precision_score(y_test, y_pred)) + ','
        data += str(metrics.recall_score(y_test, y_pred)) + ','
        data += str(metrics.f1_score(y_test, y_pred)) + ','
        fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred)
        data += str(metrics.auc(fpr, tpr)) + ','
        tn, fp, fn, tp = metrics.confusion_matrix(y_test,y_pred).ravel()
#print("True Negative: " + str(tn))
#print("False Positive: " + str(fp))
#print("False Negative: " + str(fn))
#print("True Positive: " + str(tp))
        data += str(tn) + ','
        data += str(fp) + ','
        data += str(fn) + ','
        data += str(tp)
        f_name.write(data+'\n')

