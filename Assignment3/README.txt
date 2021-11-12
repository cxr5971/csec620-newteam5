Introduction:

    In this assignment we implemented a linear SVM perform malware classification. Our SVM has two modes. The first
    is a simple linear SVM that classifies samples as either benign or malicious. THe second mode implements a One vs
    All classification model that trains a classifier for the different malware families and predicts what family of
    malware a sample belongs to. This project also includes a comparison of linear on radial SVMs using sklearn

Required modules:

   This project uses multiple external library. First the project requires the pytorch module. In order to install
   pytorch you need to be running a 64 bit version of python or else its not supported. If you are, you should be able
   install python using the following command:

       windows:
            pip3 install torch==1.10.0+cu102 torchvision==0.11.1+cu102 torchaudio===0.10.0+cu102 -f https://download.pytorch.org/whl/cu102/torch_stable.html

       Linux and Mac:
            pip3 install torch torchvision torchaudio

   If that fails their are other possible ways to install pytorch and that information can be found here:

             https://pytorch.org/get-started/locally/

   This project also uses sklearn. To install the sklearn library simply run:

             pip3 install sklearn

   This will install the necessary packages and should also install numpy if it isn't already installed. Numpy is
   another needed module to run the SVM.  If it fails to install or just isn't installed you can install numpy with:

             pip3 install numpy

   Finally this project also uses the pandas library. TO install it simply use:

             pip3 install pandas

Usage:

    Both the basic linear and the One-vs-All SVMs are implemented in SVM.py. To run the generic linear SVM simply run:

            python3 SVM.py

    Now if you want to run the One-vs-All classifier you will need to pass in the ova option at the command line. To run
    this version use:

            python3 SVM.py -ova

    This project also runs a linear and radial SVM using the sklearn's implementation to perform comparisons on the two
    kernal types. To run these SVMs simply use:

            python sklearn_SVM.py