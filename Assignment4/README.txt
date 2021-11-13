Introduction:

    In this assignment we implemented a single decision tree and a random forest classifier to determine the device
    type of different IoT devices. Once implemented, we replaced the stage1 code in classify.py with our models as well
    as added a few new command line arguments to better run the models.

Required Modules:

    Our implementation of the decision tree and Random Forest do not use any external libraries. However classify.py
    does have some requirements. These requirements can be seen in requirements.txt and can be installed by running:

        pip3 install -r {Path to}/requirements.txt

Usage:

    To run either classifier you need to run the classify.py. To run classify.py with a random forest simply run:

        python3 classify.py -r {Path to}/iot_data/

    This will cause classify.py to run and use a random forrest of 100 trees to predict the device types. If you want to
    run the program but use the decision tree classifier for the final prediction all you need to do is run:

        python3 classify.py -r {Path to}/iot_data/ -c tree

Optional Arguments:

    As stated earlier, we added new command line arguments to classify.py to provide the user more control over the
    random forest and decision trees. Here we will give a quick description of each argument but similar information
    can be received by running:

        python3 classify.py -h

    -c/--classifier - Allows you to choose the classifier. Accepts 'tree' or 'forest'. Defaults to 'forest'
    -t/--trees - Allows you to set the number of trees in the forest. Defaults to 100
    -x/--data_split - Allows you to set how much of the training data to use for each decision tree. Defaults to .5
    -f/--features - Allows you to set how many features to check try for each node when building the random forest.
                    Defaults to 7
    -d/--depth - Allows you to set the max depth of any branch on the tree. Defaults to 13
    -m/--min_nodes - Allows you to set the minimum number of samples needed to continue spiting. Defaults to 5