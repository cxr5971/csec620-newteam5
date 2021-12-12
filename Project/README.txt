Introduction:

    For our project attempted to classify netflow data as either normal or malicious traffic. To do this we
    processed the data into time windows and then used a random forest, an SVM, logistic regression, and a neural
    network to classify the samples. For all the classifiers we used the sklearn implementations of each model

Required Modules:

    This project uses 5 external libraries: pandas, numpy, sklearn, matplotlib, and scipy. All of these modules can
    simply be installed using pip3, for example:

            pip3 install sklearn

    As part of the data processing, this project saves the data to a .h5 file. This functionality is built into pandas,
    but may require you to install the dependency pytables. To install pytables try running:

            pip3 install tables

    If you are running this project with python 3.9 and on a windows machine you may run into some problems with the
    installation. If that is the case you can try installing tables using the provided wheel file
    "tables-3.6.1-cp39-cp39-win_amd64.whl". To install using the wheel you run:

            pip3 install tables-3.6.1-cp39-cp39-win_amd64.whl

    If you also happen to have an alternate version of python you can try using that version instead.

Alternate Requirements:

    To run this program you need to be using a 64 bit version of python. While the pandas library works for 32 bit
    versions of python, we were seeing memory errors when loading the data. These errors did not occur in the 64
    bit architecture

Instructions:

    The project has two main files, process_data.py and feature_extraction.py. Process_data.py is used to load the
    bidirectional netflow data and perform most of the data processing. This takes a decent amount of time, however we
    save the processed data after execution so you should not need to run this file. To run the classifiers simply use
    python to run feature_extraction.py.

            python3 feature_extraction.py

    This will run all four classifiers. If you wish to disable any of them simply comment out the associated function
    call in the main method of feature_extraction.py