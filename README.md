# FruitRecognitionProject
Machine Learning Project to recognize fruits using Kaggle's fruit dataset: https://www.kaggle.com/chrisfilo/fruit-recognition

Preprocessing
- The filenames are not consistent, as some have numbers before the fruit name and some have numbers (or pi) after.  Some of the image names are misspelled.  So I chose to use the folder names as the labels.
- I did rename some of the folders so that they were all consistent, with the first letter being uppercase.
- I also removed the folders with the "sub-categories" of fruits and moved the sub-folder with all (or most) of the fruits into the main folder.  (Many of the fruits with sub-categories have one sub-folder with most of the images and then individual sub-folders with the different fruit varieties).
- The "RenamePlum" file goes through the Plum folder and renames all the files because the files had the pi symbol in the name and weren't being loaded properly.
- I did not do any image preprocessing because the images were fairly clear, taken in a consistent location, etc.

Model 
- I experimented with a number of different models with different numbers of layers with a variety of numbers of neurons, different dropout layers, different batch size, different numbers of epochs, different learning rates and momentum.  The model in the repository is the most successful model.
- I set aside 15% of the data as testing data and then took another 15% of the training data for validation.

Further study
- The images were all fairly clear, taken in a consistent location with fairly consistent lighting, etc.  It would be interesting to add in some images that were not as clear or had different lighting and see how well the model performs
