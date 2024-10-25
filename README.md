# Simpleenergy-Hiring-Task

This repository contains the analysis and machine learning model development for the Simple Energy Hiring Challenge. The goal of this task is to analyze a provided dataset and develop a model to predict the Effective State of Charge (SOC) using various machine learning algorithms.

## Table of Contents
- [Objective](#objective)
- [Dataset](#dataset)
- [Analysis](#analysis)
- [Machine Learning Models](#machine-learning-models)
- [Key Performance Indicators (KPIs)](#key-performance-indicators-kpis)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)

## Objective
The objective of this task is to demonstrate proficiency in building machine learning models and generating key performance indicators (KPIs) from the dataset provided.

## Dataset
The dataset used in this analysis includes the following features:
- Fixed Battery Voltage
- Portable Battery Voltage
- Portable Battery Current
- Fixed Battery Current
- Motor Status (On/Off)
- BCM Battery Selected
- Portable Battery Temperatures
- Fixed Battery Temperatures

The target variable is:
- Effective SOC

## Analysis
A thorough exploration of the dataset was conducted to identify relationships between the features and the target variable. The analysis included data cleaning, preprocessing, and visualization.

## Machine Learning Models
Multiple machine learning algorithms were employed to predict the Effective SOC, including:
- Linear Regression
- Decision Tree
- Random Forest
- Gradient Boosting
- Support Vector Regressor
- K-Nearest Neighbors

The performance of each model was evaluated using R-squared and Mean Squared Error (MSE) metrics.

## Key Performance Indicators (KPIs)
KPIs were calculated to provide insights into battery performance, charge cycles, and range. The KPIs help in understanding the operational efficiency and effectiveness of the battery systems.

## Results
The results of the model evaluations are as follows:
- **Linear Regression**: R-squared: 0.9998 | MSE: 0.0009
- **Decision Tree**: R-squared: 0.8816 | MSE: 0.4660
- **Random Forest**: R-squared: 0.9719 | MSE: 0.1106
- **Gradient Boosting**: R-squared: 0.9886 | MSE: 0.0447
- **Support Vector Regressor**: R-squared: 0.9856 | MSE: 0.0566
- **K-Nearest Neighbors**: R-squared: 0.7669 | MSE: 0.9174

## Installation
To run this project locally, ensure you have Python installed along with the necessary libraries. You can install the required libraries using pip:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
