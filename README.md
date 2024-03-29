Gapminder Data Analysis Project
===============================
This repository contains a comprehensive data analysis project aimed at exploring socio-economic indicators globally, predicting future trends in key indicators, and classifying countries into different development categories based on the Gapminder dataset.

Dataset
=======
The analysis is based on the Gapminder dataset, which can be accessed here.

Dataset URL: https://raw.githubusercontent.com/BME1478H/Winter2022class/master/data/world-data-gapminder.csv

Research Questions
==================
The project focuses on answering the following pivotal questions:

Evolution of Socio-Economic Indicators: How have socio-economic indicators evolved globally over the years?
Prediction of Future Trends: Can we predict future trends in key indicators, such as Life expectancy and Income per capita?
Classification of Development Categories: What factors contribute to the classification of countries into different development categories?


Project Structure
=================
The project is structured as follows:

Data Cleaning: Initial preprocessing steps to clean the dataset.
Exploratory Data Analysis (EDA): Visual and statistical analysis to understand the data better.
Feature Selection: Identification of relevant features for the modeling process.
Modeling: Application of three machine learning models for both regression (to predict future trends) and classification (to classify countries into development categories).
Model Evaluation: Cross-validation to finalize the most accurate model among the ones used.

Required Libraries
==================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVR, SVC
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report

Steps Overview
==============
Data Loading and Cleaning
Exploratory Data Analysis (EDA)
Feature Selection
Data Preprocessing
Model Selection and Training
Regression Models: Linear Regression, Random Forest Regressor, Support Vector Regressor
Classification Models: Logistic Regression, Random Forest Classifier, Support Vector Classifier
Model Evaluation
Conclusion and Model Finalization

Conclusion
==========
This project leverages machine learning techniques to provide insights into the evolution of socio-economic indicators, predict future trends, and classify countries into development categories. Through careful data cleaning, feature selection, and model evaluation, the project aims to deliver valuable findings that contribute to the understanding of global development patterns.

