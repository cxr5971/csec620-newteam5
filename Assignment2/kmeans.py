import process_data
import random
import numpy
import statistics
import math
from matplotlib import pyplot



def k_means(training_data, normal_data, malicious_data, k_value, threshold_value):
    """
    This method provides the main k_means functionality of determining if a given data point resides within threshold_value of a centroid. If it does, it belongs within the cluster. To be more specific, this algorithm will find the closest centroid and see if the point is within threshold distance to the centroid thus classifying it as normal if close enough or malicious if not. 
    This method also performs the classification testing against the test data, and identifies important metrics (TP, FP, TN, FN, Accuracy, F1-Score, False Positive Rate, True Positive Rate)
    :param training_data: numpy.ndarray: Array of training data points, could be in any number of dimensions as determined with process_data and the PCA algorithm
    :param normal_data: numpy.ndarray: Array of non-malicious test data samples
    :param malicious_data: numpy.ndarray: Array of malicious test data samples
    :param k_value: numpy.ndarray: Number of centroids to create (identifies how many clusters)
    :param threshold_value: numpy.ndarray: Distance metric that identifies if a datapoint is close enough to be considered a part of the corresponding cluster
    :return: output which is a dictionary of the test metrics (keys: tp, fp, tn, fn, f1-score, accuracy, fpr, tpr)
    """
    centroids = numpy.array(random.sample(training_data.tolist(), k=k_value)) #select a random k centroids
    clusters, anomolies = make_cluster(centroids, training_data, threshold_value) #cluster the training data based on the centroids
    new_centroids = list()
    for cluster in clusters:
        new_centroids.append(calculate_mean_of_cluster(cluster)) #calculate new centroids based on the mean of the training data

    #Classifying the good test data, anything that is within the centroid distance is classified
    # as within a cluster and therefore normal data. Anything that isn't within centroid is
    # classified as bad and thus a bad classification
    good_bois = []
    bad_bois = []
    for point in normal_data:
        distance = 99999
        for centroid in new_centroids:
            point_centroid_dist = numpy.linalg.norm(point - centroid) 
            if point_centroid_dist <= distance:
                distance = point_centroid_dist
        if distance <= threshold_value:
            good_bois.append(point)
        else:
            bad_bois.append(point)
    
    false_positive = len(bad_bois)
    true_negative = len(good_bois)
    


    #Classifying the bad test data, anything that is within the centroid distance is classified
    # as within a cluster and therefore normal data this is a false negative. Anything that isn't within centroid is classified as bad and thus a true positive
    good_bois = []
    bad_bois = [] 
    for point in malicious_data:
        distance = 99999
        for centroid in new_centroids:
            point_centroid_dist = numpy.linalg.norm(point - centroid) 
            if point_centroid_dist <= distance:
                distance = point_centroid_dist
            
        if distance <= threshold_value:    
            good_bois.append(point)
        else:
            bad_bois.append(point)

    false_negative = len(good_bois)
    true_positive = len(bad_bois)
    accuracy = (true_positive + true_negative) / (true_positive + false_positive + true_negative + false_negative)
    TPR = true_positive / (false_negative + true_positive)
    FPR = false_positive / (true_negative + false_positive) 
    F1_score = (2*(true_positive) / (2*(true_positive) + false_positive + false_negative))

    #Print statements left commented out for now
    print("K: " + str(k_value))
    print("Threshold: " + str(threshold_value))
    print("TP: " + str(true_positive))
    print("FN: " + str(false_negative))
    print("TN: " + str(true_negative))
    print("FP: " + str(false_positive))
    print("Accuracy: " + str(accuracy))
    print("TPR: " + str(TPR))
    print("FPR: " + str(FPR))
    print("F1-score: "+ str(F1_score))


    #Saving output to be used elsewhere, this is used in hyperparameter tuning
    output = {}
    output['tp'] = true_positive
    output['fn'] = false_negative
    output['tn'] = true_negative
    output['fp'] = false_positive
    output['accuracy'] = accuracy
    output['tpr'] = TPR
    output['fpr'] = FPR
    output['f1-score'] = F1_score
    return output


        
def make_cluster(centroids, data, threshold):
    """
    Clusters a set of data using provided centroids with a user provided threshold
    :param centroids: a list of centroids to cluster with
    :param data: the data to cluster
    :param threshold: the threshold to use.  zero for none.
    :ret a list of clusters and a list of items that did not fit into the clusters (anomolies)
    """
    clusters = list() # initialize the clusters
    for x in range(len(centroids)):
        clusters.append(list())
    anomolies = list() # create something to store the anomolies in.
    for item in data:
        distances = list()
        for x in range(len(centroids)):
            distance = numpy.linalg.norm(item - centroids[x]) #calculate the distance of the item from the centroid
            if threshold == 0 or distance < threshold:  #determine if this is within the threshold, if needed.
                distances.append((x, distance))
        
        if len(distances) == 0:
            anomolies.append(item) #no distances were within the threshold.  Its an anomoly.
        else:
            #find the nearest cluster
            def sort_func(data):
                return data[1]
            distances.sort(key=sort_func)
            clusters[distances[0][0]].append(item) #find the cluster that the item is closest to, and append to that cluster.
    return clusters, anomolies


def calculate_mean_of_cluster(cluster):
    """
    Find the mean value of a cluster, indicating it's relative center
    :param cluster: list: set of points that make up a cluster
    :ret: coordinate that represents the mean 
    """
    total_point = len(cluster)
    features = cluster[0].shape[0]
    coordinates = [0] * features
    for point in cluster:
        coordinates = [a + b for a, b in zip(coordinates, point)]

    for i in range(len(coordinates)):
        coordinates[i] = coordinates[i] / total_point
    return coordinates



def hype_test(min_k, max_k, min_thresh, max_thresh, training_data, testing_normal, testing_malicious, dimensions):
    """
    This function will perform hyper parameter testing of kmeans. It is essentially a big nested for loop that keeps running k_means and interpreting its output with a range of k and threshold values to use. It outputs this data in a scatter plot.
    :param min_k: int: Used to indicate the lowest k value to use for k_means
    :param max_k: int: Used to indicate the highest k value to use for k_means
    :param min_thresh: int: Used to indicate the lowest threshold value to use for k_means
    :param max_thresh: int: Used to indicate the highest threshold value to use for k_means
    :param training_data: numpy.ndarray: Training data numpy array to use for k_means
    :param testing_normal: numpy.ndarray: testing data that is normal traffic
    :param testing_malicious: numpy.ndarray: testing data that is malicious
    :param dimensions: int: number of dimensions to represent the data in
    :return: Nothing but the scatter plot being shown
    """ 
    test_output = {}
    for i in range(min_k, max_k):
        #The numpy.arange function will help us loop through exection, change .005 identifies the step value for epsilon
        for j in numpy.arange(min_thresh,max_thresh,.005):
            k_value = round(i,3)
            thresh_value = round(j,3)
            key = str(k_value) + ":" + str(thresh_value)
            test_output[key] = k_means(training_data, testing_normal, testing_malicious, k_value, thresh_value)


    #This section creates a scatter plot of F1-Score performance based on the parameters chosen for hype_test
    scores = []
    k_list = []
    thresh_list = []
    for key in test_output:
        k_value = key.split(":")[0]
        thresh_value = key.split(":")[1]
        k_list.append(k_value)
        thresh_list.append(thresh_value)
        scores.append(test_output[key]['f1-score'])
    some_var = pyplot.figure() #Creating a figure that will become the colorbar
    cm = pyplot.cm.get_cmap('copper') 
    sc = pyplot.scatter(k_list,thresh_list,c=scores,vmin=0.5,vmax=1,cmap=cm)
    cb = some_var.colorbar(sc) #Turning figure into colorbar
    cb.set_label("F1-Score")
    pyplot.xlabel("K")
    pyplot.ylabel("Threshold")
    pyplot.title("K-Means Hyperparameter Effectiveness with " + str(dimensions) + " features")
    pyplot.show()
    
        



def main():
    #Set the list of dimensions to use
    dimensions = 41
    testing_normal, testing_malicious, training_data = process_data.prepare_data(dimensions)
    print("Dimensions: " + str(dimensions))
    k_means(training_data, testing_normal, testing_malicious, 3, 0.29)
    
    #Line below is an example hyperparameter testing function call (testing between 2 and 11 for k values, between 0.03 and 0.2 for epsilon
    #hype_test(2,11,0.03,0.2,training_data, testing_normal, testing_malicious,dimensions)

main()
