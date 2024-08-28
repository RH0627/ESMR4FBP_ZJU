# ESMR4FBP_ZJU
If you only need to predict antioxidant peptides or bitter peptides, you only need to completely download the app_development folder and install the relevant environment to use it!

environment requested

pip install fair-esm

pip install tensorflow

pip install pytorch

pip install numpy 

pip install pandas


Given the conflict between the pytorch environment and the Tensorflow environment, we separated the embedding part from the prediction part. 
However, this does not affect the usage and you do not need to have any knowledge about ESM-2. 
All you need to do is enter the sequences or files as required and the program will automatically do the peptide embedding for you. 
You can load the peptide embedded file into prediction_app for prediction.

