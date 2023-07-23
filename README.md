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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import gower
from prince import MCA
import prince
from sklearn import datasets, cluster
from sklearn.cluster import DBSCAN, KMeans
from sklearn import random_projection
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

## File descriptions 
Projekt4.ipynb - Jupyter Notebook which contains the code and results 
output folder - mostly text files with statistics, sometimes figures. No executable code, only results. 
## Acknowledgements
To Arvato Financial Services for providing the data,
to Udacity for teaching the skills,
to the Python community who asks and answers so many questions that I was always able to relate to one when I encountered a problem. 


