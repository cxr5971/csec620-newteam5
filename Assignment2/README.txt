Introduction:

        In this project we implemented the K-means and DBSCAN algorithms to perform anomaly detection on the provided
        KDD99 files. After training the algorithms, any new point that lands in a cluster is assumed to be normal while
        any point that doesn't is an anomaly

Usage:

        Our project uses the sklearn library to perform PCA on the initial data as well as to tune the hyperparameters
        for our DBSCAN implementation. If you do not have sklearn installed you can install it by running:

                    pip install sklearn

        To create a scatter plot of a random sub-sample of 500 data points from the testing-normal and testing-malicious
        data sets simply run process_data.py. You can do this by running:

                    python3 process_data.py

        To run our DBSCAN algorithm you need to run the dbscan.py file. This will run the algorithm on data with 41
        dimensions, a min neighbor count of 5, and an epsilon value of .06. To run the clustering algorithm simply run:

                    python3 dbscan.py

        To run our implementation of K-means you will need to run the kmeans.py file. This will run the algorithm with
        a K of 3 and a threshold of .005 on 2 dimensional data. To run the algorithm simply run:

                    python3 kmeans.py