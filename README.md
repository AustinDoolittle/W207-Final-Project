# W207 Summer 2019 Final Project
Amy Breden, Austin Doolittle, Ellie Huang, Ray Xiao

## Summary
For our final project, we attempted the [Zillow Home Value Prediction Kaggle competition](https://www.kaggle.com/c/zillow-prize-1). The goal of the competition is to predict the log error of the Zillow estimate (Zestimate), as calculated below:

logerror = log(Sale Price) - log(Zestimate)

The competition scores entries based on the Mean Absolute Error or MAE.

## Datasets
The datasets are large and should be downloaded [directly from Kaggle](https://www.kaggle.com/c/zillow-prize-1/data) or using Kaggle's API:

`kaggle competitions download -c zillow-prize-1`

## File structure
Our analysis and modeling is in combinded_notebook.ipynb. We used the json files to indicate which features to include in each dataset and how to populate NaN values. 
