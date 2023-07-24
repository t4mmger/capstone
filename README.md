# Udacity Capstone Project

## Project Overview
This project is the final part of the Data Science course from Udacity. 
The task is to analyze demographics data for customers of a mail-order sales company in Germany, comparing it against demographics information for the general population. We perform customer segmentation, identifying the parts of the population that best describe the core customer base of the company. 
Then, we apply what we learned from the segmentation on a third dataset with demographics information for targets of a marketing campaign for the company, and use a model to predict which individuals are most likely to convert into becoming customers for the company.

## Motivation for Project
My motivation why I chose this project was that you can apply all skills that are need to be a successfull data scientist:
- data wrangling
- unsupervised learning
- supervised learning

## How I approached the problem
data wrangling was about treatment of missing values, dropping rows and columns with too many missing values, identifying outliers.
unsupervised learning: I used Elbow method to get a good indication of the number of clusters and then used k-means to assign the observations to clusters. Comparisons between customers and population were done with graphics and statistics. For categorical features I chose DBSCAN and plotted them because they were only 2D. In both cases I was able to identify variables that can be used to identify (future) customers. 
supervised learning: This is still open and will be done the next days. 


## Summary of results: 
- I identified differences between the customers and the sample which was representative for Germany.
- This will be useful for a future mailing campaign. 

## Libraries
import numpy as np \n
import pandas as pd \n
import matplotlib.pyplot as plt \n
import seaborn as sns \n
import gower \n
from prince import MCA \n
from sklearn.cluster import KMeans, DBSCAN \n
from sklearn import datasets, cluster \n
from sklearn import random_projection \n
from sklearn.decomposition import PCA \n
from sklearn.model_selection import train_test_split, GridSearchCV \n
from sklearn.preprocessing import StandardScaler \n
from sklearn.metrics import mean_absolute_error, roc_curve, roc_auc_score \n
from sklearn.linear_model import LogisticRegression \n
from sklearn.pipeline import Pipeline \n

import warnings
from xgboost import XGBRegressor

## File descriptions 
Projekt4.ipynb - Jupyter Notebook which contains the code and results 
output folder - mostly text files with statistics, sometimes figures. No executable code, only results. 
## Acknowledgements
To Arvato Financial Services for providing the data,
to Udacity for teaching the skills,
to the Python community who asks and answers so many questions that I was always able to relate to one when I encountered a problem. 


