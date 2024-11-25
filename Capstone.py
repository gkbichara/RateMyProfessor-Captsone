#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 13:42:07 2024

@author: Galal Bichara
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

noNanNumericalDf = numericalDf.dropna(subset=['AvgRating'])


threshold = noNanNumericalDf['numRatings'].quantile(0.5)



# %% Question 1


# EDA
avgRatingBins = np.arange(0, 5.5, 0.5)  # Creating bins from 0 to 5 with a step of 0.5
plt.figure(figsize=(10, 6))
plt.hist(noNanNumericalDf['AvgRating'], bins=avgRatingBins, edgecolor='black', alpha=0.7)
plt.title('Distribution of Average Ratings')
plt.xlabel('Average Rating')
plt.ylabel('Frequency')
plt.xticks(avgRatingBins)  # Setting x-ticks to match the bin edges for clarity
plt.show()


#  Improved Visualization of Distribution of Number of Ratings
plt.figure(figsize=(10, 6))
# Use logarithmic bins for better visual representation of skewed data
ratingsBins = np.concatenate([np.arange(1, 51, 2), np.arange(51, 401, 25)])  # Finer bins in the lower range
plt.hist(noNanNumericalDf['numRatings'], bins=ratingsBins, edgecolor='black', alpha=0.7)
plt.title('Distribution of Average Ratings')
plt.xlabel('Number of Ratings')
plt.ylabel('Frequency')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.yscale('log')  # Set the y-axis to log scale for better visibility of all values
plt.show()



# Filter dataset to only include professors with >= threshold number of ratings
q1Df = noNanNumericalDf[noNanNumericalDf['numRatings'] > threshold]

femaleProfs = q1Df[q1Df["Female"] == 1]
maleProfs = q1Df[q1Df["Male"] == 1]
maleProfRatings = maleProfs["AvgRating"]
femaleProfRatings = femaleProfs["AvgRating"]

q1u, q1p = stats.mannwhitneyu(maleProfRatings, femaleProfRatings)



avgRatingBins = np.arange(0, 5.5, 0.5)  # Creating bins from 0 to 5 with a step of 0.5
plt.figure(figsize=(10, 6))
plt.hist(femaleProfs['AvgRating'], bins=avgRatingBins, edgecolor='black', alpha=0.7)
plt.title('Distribution of Female Ratings')
plt.xlabel('Average Rating')
plt.ylabel('Frequency')
plt.xticks(avgRatingBins)  # Setting x-ticks to match the bin edges for clarity
plt.show()


avgRatingBins = np.arange(0, 5.5, 0.5)  # Creating bins from 0 to 5 with a step of 0.5
plt.figure(figsize=(10, 6))
plt.hist(maleProfs['AvgRating'], bins=avgRatingBins, edgecolor='black', alpha=0.7)
plt.title('Distribution of Male Ratings')
plt.xlabel('Average Rating')
plt.ylabel('Frequency')
plt.xticks(avgRatingBins)  # Setting x-ticks to match the bin edges for clarity
plt.show()


# %% Question 2







# %% Question 3

from scipy.stats import spearmanr


q3df = noNanNumericalDf[noNanNumericalDf['numRatings'] > 15]


avgRatingArray = q3df["AvgRating"].values  
avgDifficultyArray = q3df["AvgDifficulty"].values  


valid_mask = ~np.isnan(avgRatingArray) & ~np.isnan(avgDifficultyArray)


avgRatingArray_clean = avgRatingArray[valid_mask]
avgDifficultyArray_clean = avgDifficultyArray[valid_mask]


plt.figure(figsize=(10, 6))
plt.scatter(avgRatingArray_clean, avgDifficultyArray_clean, alpha=0.7)
plt.xlabel('Average Rating')
plt.ylabel('Average Difficulty')
plt.title('Average Rating vs Average Difficulty (Cleaned Data)')
plt.grid(axis='both', linestyle='--', alpha=0.7)
plt.show()

rho = spearmanr(avgRatingArray_clean, avgDifficultyArray_clean)






# %% Question 5


noNanPropTakeAgainDf = numericalDf.dropna(subset=['propTakeAgain'])
nonNanPropWithThreshold = noNanPropTakeAgainDf[noNanPropTakeAgainDf["numRatings"] > 3]
# can do without imputation, have 12k ratings, Circle back to it
