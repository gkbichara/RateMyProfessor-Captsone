#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 13:42:07 2024

@author: Galal BIchara
"""
# %% Loading in all the libraries and datasets
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy import stats
import pandas as pd


numericalDf = pd.read_csv("/Users/gkb/Desktop/PODS/Capstone/rmpCapstoneNum.csv", header = None, 
           names = ["AvgRating","AvgDifficulty", "numRatings", "Hot", "propTakeAgain", "numRatingsOnline", 
           "Male", "Female"])

qualitativeDf = pd.read_csv("/Users/gkb/Desktop/PODS/Capstone/rmpCapstoneQual.csv", header = None, 
            names = ["Major/Field", "University", "State"])

noNanAvgRatings = numericalDf.dropna(subset=['AvgRating']).shape[0]


threshold = 5



# %% Question 1


# EDA
avgRatingBins = np.arange(0, 5.5, 0.5)  # Creating bins from 0 to 5 with a step of 0.5
plt.figure(figsize=(10, 6))
plt.hist(numericalDf['AvgRating'], bins=avgRatingBins, edgecolor='black', alpha=0.7)
plt.title('Distribution of Average Ratings')
plt.xlabel('Average Rating')
plt.ylabel('Frequency')
plt.xticks(avgRatingBins)  # Setting x-ticks to match the bin edges for clarity
plt.show()


#  Improved Visualization of Distribution of Number of Ratings
plt.figure(figsize=(10, 6))
# Use logarithmic bins for better visual representation of skewed data
ratingsBins = np.concatenate([np.arange(1, 51, 2), np.arange(51, 401, 25)])  # Finer bins in the lower range
plt.hist(numericalDf['numRatings'], bins=ratingsBins, edgecolor='black', alpha=0.7)
plt.title('Distribution of Number of Ratings')
plt.xlabel('Number of Ratings')
plt.ylabel('Frequency')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.yscale('log')  # Set the y-axis to log scale for better visibility of all values
plt.show()






# Filter dataset to only include professors with >= threshold number of ratings
filteredDf = numericalDf[numericalDf['numRatings'] >= threshold]





# %% Question 2







# %% Question 3

avgRatingArray = np.array([filteredDf["AvgRating"]])
avgDifficultyArray = np.array([filteredDf["AvgDifficulty"]])

question3nans = np.array([np.isnan(avgRatingArray), np.isnan(avgDifficultyArray)], dtype = bool)

# same info but in numbers
question3nans2 = question3nans * 1

# summing tells you how many movies each person is missing (0 means they rated all 3)
question3nans3 = sum(question3nans2)

missingData = np.where(question3nans3 > 0)

# delete all users that have not rated all 3 movies
# make sure you delete same users from all 3
avgRatingArray = np.delete(avgRatingArray, missingData)
avgDifficultyArray = np.delete(avgDifficultyArray, missingData)

plt.scatter(avgRatingArray, avgDifficultyArray)

