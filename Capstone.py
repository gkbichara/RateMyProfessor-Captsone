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
import random
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns
from sklearn.linear_model import Ridge
import statsmodels.api as sm



SEED = 16906324
random.seed(SEED)
np.random.seed(SEED)


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



# Filter dataset to only include professors with > threshold number of ratings
q1Df = noNanNumericalDf[noNanNumericalDf['numRatings'] > threshold]

femaleProfs = q1Df[q1Df["Female"] == 1]
maleProfs = q1Df[q1Df["Male"] == 1]
maleProfRatings = maleProfs["AvgRating"]
femaleProfRatings = femaleProfs["AvgRating"]

q1u, q1p = stats.mannwhitneyu(maleProfRatings, femaleProfRatings)



avgRatingBins = np.arange(0, 5.5, 0.5)  # Creating bins from 0 to 5 with a step of 0.5

plt.figure(figsize=(10, 6))

# Plot histogram for female professors
plt.hist(femaleProfs['AvgRating'], bins=avgRatingBins, alpha=0.5, label='Female Professors', color='yellow', edgecolor='black')

# Plot histogram for male professors
plt.hist(maleProfs['AvgRating'], bins=avgRatingBins, alpha=0.5, label='Male Professors', color='blue', edgecolor='black')

# Adding labels and title
plt.xlabel('Average Rating')
plt.ylabel('Frequency')
plt.title('Distribution of Average Ratings by Gender')
plt.xticks(avgRatingBins)
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.show()


q1maleProfsMean = maleProfRatings.mean()
q1femaleProfsMean = femaleProfRatings.mean()
q1std_combined = np.sqrt(((maleProfRatings.std() ** 2) + (femaleProfRatings.std() ** 2)) / 2)
q1cohens_d = (q1maleProfsMean - q1femaleProfsMean) / q1std_combined
print(q1cohens_d)


male_median = maleProfs['AvgRating'].median()
female_median = femaleProfs['AvgRating'].median()

boxplot_data = q1Df[['AvgRating', 'Male', 'Female']].copy()
boxplot_data['Gender'] = boxplot_data.apply(lambda x: 'Male' if x['Male'] == 1 else 'Female', axis=1)

plt.figure(figsize=(10, 6))
sns.boxplot(x='Gender', y='AvgRating', data=boxplot_data, palette='Set3', width=0.5)
plt.title('Box Plot of Average Ratings by Gender')
plt.xlabel('Gender')
plt.ylabel('Average Rating')
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.show()


# %% Question 2


q2Threshold = q1Df['numRatings'].quantile(0.75)

inexperiencedProfs = q1Df[q1Df['numRatings'] <= q2Threshold]
experiencedProfs = q1Df[q1Df['numRatings'] > q2Threshold]



plt.hist(experiencedProfs['AvgRating'], bins=avgRatingBins, alpha=0.5, label='Experienced Professors', color='blue', edgecolor='black')

# Plot histogram for inexperienced professors
plt.hist(inexperiencedProfs['AvgRating'], bins=avgRatingBins, alpha=0.5, label='Inexperienced Professors', color='yellow', edgecolor='black')

# Adding labels, title, and legend
plt.xlabel('Average Rating')
plt.ylabel('Frequency')
plt.title('Distribution of Average Ratings: Experienced vs Inexperienced Professors')
plt.xticks(avgRatingBins)
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)





experienced_ratings = experiencedProfs['AvgRating']
inexperienced_ratings = inexperiencedProfs['AvgRating']



q2u, q2p = stats.mannwhitneyu(experienced_ratings, inexperienced_ratings)

q2experiencedProfsMean = experienced_ratings.mean()
q2inexperiencedProfsMean = inexperienced_ratings.mean()
q2std_combined = np.sqrt(((experienced_ratings.std() ** 2) + (inexperienced_ratings.std() ** 2)) / 2)
q2cohens_d = (q2experiencedProfsMean - q2inexperiencedProfsMean) / q2std_combined
print(q2cohens_d)


experienced_median = experiencedProfs['AvgRating'].median()
inexperienced_median = inexperiencedProfs['AvgRating'].median()

boxplot_data_q2 = q1Df[['AvgRating', 'numRatings']].copy()
boxplot_data_q2['Experience'] = boxplot_data_q2['numRatings'].apply(lambda x: 'Inexperienced' if x <= q2Threshold else 'Experienced')

# Create the box plot
plt.figure(figsize=(10, 6))
sns.boxplot(x='Experience', y='AvgRating', data=boxplot_data_q2, palette='Set3', width=0.5)
plt.title('Box Plot of Average Ratings: Experienced vs Inexperienced Professors')
plt.xlabel('Experience Level')
plt.ylabel('Average Rating')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()


# %% Question 3

from scipy.stats import spearmanr

q3Threshold = q1Df['numRatings'].quantile(0.9)

q3df = noNanNumericalDf[noNanNumericalDf['numRatings'] > q3Threshold]


avgRatingQ3Array = q3df["AvgRating"].values  
avgDifficultyArray = q3df["AvgDifficulty"].values  

# find all rows that don't have nan in average rating and difficulty
valid_mask = ~np.isnan(avgRatingQ3Array) & ~np.isnan(avgDifficultyArray)


avgRatingQ3Array_clean = avgRatingQ3Array[valid_mask]
avgDifficultyArray_clean = avgDifficultyArray[valid_mask]


plt.figure(figsize=(10, 6))
plt.scatter(avgDifficultyArray_clean, avgRatingQ3Array_clean, alpha=0.7)
plt.xlabel('Average Difficulty')
plt.ylabel('Average Rating')
plt.title('Average Difficulty vs Average Rating (Cleaned Data)')
plt.grid(axis='both', linestyle='--', alpha=0.7)
plt.show()

q3rho, q3p = spearmanr(avgDifficultyArray_clean, avgRatingQ3Array_clean)


avgDifficultyArray_clean_reshaped = avgDifficultyArray_clean.reshape(-1, 1)

# Create and fit the model
reg_model = LinearRegression()
reg_model.fit(avgDifficultyArray_clean_reshaped, avgRatingQ3Array_clean)

# Predict the ratings based on the difficulty values
q3predicted_ratings = reg_model.predict(avgDifficultyArray_clean_reshaped)

# Calculate residuals
q3residuals = avgRatingQ3Array_clean - q3predicted_ratings

# Plot the residuals with predicted ratings on x-axis
plt.figure(figsize=(10, 6))
plt.scatter(q3predicted_ratings, q3residuals, alpha=0.7, color='purple', edgecolor='black')
plt.axhline(y=0, color='r', linestyle='--')  # Add a horizontal line at y=0
plt.xlabel('Predicted Average Rating')
plt.ylabel('Residuals (Actual Rating - Predicted Rating)')
plt.title('Residual Plot: Predicted Average Rating vs Residuals')
plt.grid(axis='both', linestyle='--', alpha=0.7)
plt.show()

plt.figure(figsize=(10, 6))
sm.qqplot(q3residuals, line='45', fit=True)
plt.title("Q-Q Plot of Residuals")
plt.grid(axis='both', linestyle='--', alpha=0.7)
plt.show()


# %% Question 4


q4df = q1Df.dropna(subset=['numRatingsOnline'])

# Split the filtered dataset into two groups based on the percentage of online ratings
onlineProfessors = q4df[q4df['numRatingsOnline'] >= (0.5 * q4df['numRatings'])]
offlineProfessors = q4df[q4df['numRatingsOnline'] == 0]


online_ratings = onlineProfessors['AvgRating']
offline_ratings = offlineProfessors['AvgRating']



q4u, q4p = stats.mannwhitneyu(online_ratings, offline_ratings)
print(q4p)

q4onlineProfsMean = online_ratings.mean()
q4offlineProfsMean = offline_ratings.mean()
q4std_combined = np.sqrt(((online_ratings.std() ** 2) + (offline_ratings.std() ** 2)) / 2)
q4cohens_d = (q4onlineProfsMean - q4offlineProfsMean) / q4std_combined
print(q4cohens_d)


online_median = onlineProfessors['AvgRating'].median()
offline_median = offlineProfessors['AvgRating'].median()



# %% Question 5


noNanPropTakeAgainDf = numericalDf.dropna(subset=['propTakeAgain'])

q5Threshold = noNanPropTakeAgainDf['numRatings'].quantile(0.75)
nonNanPropWithThreshold = noNanPropTakeAgainDf[noNanPropTakeAgainDf["numRatings"] > q5Threshold]



avgRatingQ5Array = nonNanPropWithThreshold["AvgRating"].values  
propTakeAgainArray = nonNanPropWithThreshold["propTakeAgain"].values  


valid_mask = ~np.isnan(avgRatingQ5Array) & ~np.isnan(propTakeAgainArray)


avgRatingQ5Array_clean = avgRatingQ5Array[valid_mask]
propTakeAgainArray_clean = propTakeAgainArray[valid_mask]


plt.figure(figsize=(10, 6))
plt.scatter(propTakeAgainArray_clean, avgRatingQ5Array_clean, alpha=0.7)
plt.xlabel('Prop Take Again')
plt.ylabel('Average Rating')
plt.title(' Prop Take Again vs Average Rating (Cleaned Data)')
plt.grid(axis='both', linestyle='--', alpha=0.7)
plt.show()


q5rho, q5p = spearmanr(avgRatingQ5Array_clean, propTakeAgainArray_clean)

print(q5rho)


# Fit a linear regression model
propTakeAgainArray_clean_reshaped = propTakeAgainArray_clean.reshape(-1, 1)

# Create and fit the linear regression model
reg_model = LinearRegression()
reg_model.fit(propTakeAgainArray_clean_reshaped, avgRatingQ5Array_clean)

# Predict the ratings based on the "Prop Take Again" values
predicted_ratings = reg_model.predict(propTakeAgainArray_clean_reshaped)

# Calculate residuals
residuals = avgRatingQ5Array_clean - predicted_ratings

# Plot the residuals against the predicted average ratings
plt.figure(figsize=(10, 6))
plt.scatter(predicted_ratings, residuals, alpha=0.7, color='purple', edgecolor='black')
plt.axhline(y=0, color='r', linestyle='--')  # Add a horizontal line at y=0
plt.xlabel('Predicted Average Rating')
plt.ylabel('Residuals (Actual Rating - Predicted Rating)')
plt.title('Residual Plot: Predicted Average Rating vs Residuals')
plt.grid(axis='both', linestyle='--', alpha=0.7)
plt.show()

plt.figure(figsize=(10, 6))
stats.probplot(residuals, dist="norm", plot=plt)
plt.title("Q-Q Plot of Residuals")
plt.grid(axis='both', linestyle='--', alpha=0.7)
plt.show()



# %% Question 6

# EDA
hotProfs = q1Df[q1Df["Hot"] == 1]
notHotProfs = q1Df[q1Df["Hot"] == 0]
hotProfRatings = hotProfs["AvgRating"]
notHotProfRatings = notHotProfs["AvgRating"]

# Create bins for the histograms
avgRatingBins = np.arange(0, 5.5, 0.5)  # Creating bins from 0 to 5 with a step of 0.5

# Plot both histograms on the same figure
plt.figure(figsize=(10, 6))

# Histogram for Not Hot Professors
plt.hist(notHotProfRatings, bins=avgRatingBins, edgecolor='black', alpha=0.5, label='Not Hot Professors', color='blue')

# Histogram for Hot Professors
plt.hist(hotProfRatings, bins=avgRatingBins, edgecolor='black', alpha=0.5, label='Hot Professors', color='red')

# Adding labels, title, and legend
plt.title('Distribution of Average Ratings: Hot vs Not Hot Professors')
plt.xlabel('Average Rating')
plt.ylabel('Frequency')
plt.xticks(avgRatingBins)  # Setting x-ticks to match the bin edges for clarity
plt.legend()  # Add a legend to distinguish between Hot and Not Hot professors
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Show the combined plot
plt.show()
q6u, q6p = stats.mannwhitneyu(hotProfRatings, notHotProfRatings)

q6hotProfsMean = hotProfRatings.mean()
q6notHotProfsMean = notHotProfRatings.mean()
q6std_combined = np.sqrt(((hotProfRatings.std() ** 2) + (notHotProfRatings.std() ** 2)) / 2)
q6cohens_d = (q6hotProfsMean - q6notHotProfsMean) / q6std_combined
print(q6cohens_d)



# %% Question 7


q7df = noNanNumericalDf[noNanNumericalDf['numRatings'] > 18]


avgRatingQ7Array = q7df["AvgRating"].values  
avgDifficultyQ7Array = q7df["AvgDifficulty"].values  

# find all rows that don't have nan in average rating and difficulty
valid_mask = ~np.isnan(avgRatingQ7Array) & ~np.isnan(avgDifficultyQ7Array)


q7x = avgDifficultyQ7Array[valid_mask].reshape(-1, 1)
q7y= avgRatingQ7Array[valid_mask]


q7x_train, q7x_test, q7y_train, q7y_test = train_test_split(q7x, q7y, test_size=0.2, random_state=SEED)

# Initialize the regression model
q7reg_model = LinearRegression()

# Train the model on the training data
q7reg_model.fit(q7x_train, q7y_train)

# Predict the average rating on the test data
q7y_pred = q7reg_model.predict(q7x_test)

# Calculate R^2 and RMSE
q7r2 = r2_score(q7y_test, q7y_pred)
q7rmse = np.sqrt(mean_squared_error(q7y_test, q7y_pred))

# Scatter plot of actual vs predicted ratings
plt.figure(figsize=(10, 6))
plt.scatter(q7x_test, q7y_test, alpha=0.7, label='Actual Ratings')
plt.scatter(q7x_test, q7y_pred, alpha=0.7, color='red', label='Predicted Ratings', marker='x')

# Plotting the line of best fit
x_range = np.linspace(q7x_test.min(), q7x_test.max(), 100).reshape(-1, 1)
y_range_pred = q7reg_model.predict(x_range)
plt.plot(x_range, y_range_pred, color='blue', linestyle='--', label='Line of Best Fit')

plt.xlabel('AvgDifficulty')
plt.ylabel('AvgRating')
plt.title('Actual vs Predicted Average Rating with Line of Best Fit')
plt.grid(axis='both', linestyle='--', alpha=0.7)
plt.legend()
plt.show()



q7intercept = q7reg_model.intercept_
q7coefficient = q7reg_model.coef_[0]

print(f"Intercept: {q7intercept}")
print(f"Coefficient (AvgDifficulty): {q7coefficient}")




# %% Question 8


# Filter out all rows with missing values in any column to ensure complete data
q8Df = q1Df.dropna()

# Features and target for regression
q8X = q8Df.drop(columns=['AvgRating'])  # All available features except the target variable
q8y = q8Df['AvgRating']

# Train-test split
q8X_train, q8X_test, q8y_train, q8y_test = train_test_split(q8X, q8y, test_size=0.2, random_state=SEED)

# Initialize and fit the regression model
q8reg_model = Ridge(alpha=11.0)
q8reg_model.fit(q8X_train, q8y_train)

# Predict on the test set
q8y_pred = q8reg_model.predict(q8X_test)

# Calculate R^2 and RMSE
q8r2 = r2_score(q8y_test, q8y_pred)
q8rmse = np.sqrt(mean_squared_error(q8y_test, q8y_pred))

# Print results
print(f"R^2: {q8r2}")
print(f"RMSE: {q8rmse}")

# Print individual beta q8coefficients for each feature
q8coefficients = pd.Series(q8reg_model.coef_, index=q8X.columns)
print("\nCoefficients:")
print(q8coefficients)

# Visualization: Actual vs Predicted Ratings
plt.figure(figsize=(10, 6))
plt.scatter(q8y_pred, q8y_test, alpha=0.7)
plt.xlabel('Predicted AvgRating')
plt.ylabel('Actual AvgRating')
plt.title('Actual vs Predicted Average Rating with Line of Best Fit')
plt.grid(axis='both', linestyle='--', alpha=0.7)
plt.plot([q8y.min(), q8y.max()], [q8y.min(), q8y.max()], color='red', linestyle='--', label='Perfect Fit Line')  # Line of perfect prediction
plt.legend()
plt.show()




