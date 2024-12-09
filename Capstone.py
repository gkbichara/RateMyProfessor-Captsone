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
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns
from sklearn.linear_model import Ridge
import statsmodels.api as sm
import statsmodels.stats.power as smp
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, accuracy_score
from sklearn.linear_model import RidgeCV






alpha = 0.005
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


male_median = maleProfs['AvgRating'].median()
female_median = femaleProfs['AvgRating'].median()

boxplot_data_q1 = q1Df[['AvgRating', 'Male', 'Female']].copy()
boxplot_data_q1['Gender'] = boxplot_data_q1.apply(lambda x: 'Male' if x['Male'] == 1 else 'Female', axis=1)

plt.figure(figsize=(10, 6))
sns.boxplot(x='Gender', y='AvgRating', data=boxplot_data_q1, palette='Set3', width=0.5)
plt.title('Box Plot of Average Ratings by Gender')
plt.xlabel('Gender')
plt.ylabel('Average Rating')
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.show()


# Create a separate DataFrame for regression analysis
q1RegressionDf = q1Df.copy()

# Gender encoding: 1 for Male, 0 for Female
q1RegressionDf['q1Gender'] = q1RegressionDf.apply(lambda x: 1 if x['Male'] == 1 else 0, axis=1)

# Independent variables: Gender and Number of Ratings
q1X = q1RegressionDf[['q1Gender', 'numRatings']]

# Add a constant for the intercept term
q1X = sm.add_constant(q1X)

# Dependent variable: Average Rating
q1y = q1RegressionDf['AvgRating']

# Split the data into training and test sets (80% training, 20% test)
q1X_train, q1X_test, q1y_train, q1y_test = train_test_split(q1X, q1y, test_size=0.2, random_state=SEED)

# Fit the OLS model using statsmodels
q1Model_ols = sm.OLS(q1y_train, q1X_train).fit()

# Predict on the test set
q1y_pred_ols = q1Model_ols.predict(q1X_test)

# Print the summary of the model, which includes coefficients, p-values, and other statistics
print(q1Model_ols.summary())



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

q3Threshold = noNanNumericalDf['numRatings'].quantile(0.9)

q3df = noNanNumericalDf[noNanNumericalDf['numRatings'] > 3]


avgRatingQ3Array = q3df["AvgRating"].values  
avgDifficultyArray = q3df["AvgDifficulty"].values  

# find all rows that don't have nan in average rating and difficulty
valid_mask = ~np.isnan(avgRatingQ3Array) & ~np.isnan(avgDifficultyArray)


avgRatingQ3Array_clean = avgRatingQ3Array[valid_mask]
avgDifficultyArray_clean = avgDifficultyArray[valid_mask]


plt.figure(figsize=(14, 9))
plt.scatter(avgDifficultyArray_clean, avgRatingQ3Array_clean, alpha=0.7, s=60)
plt.xlabel('Average Difficulty', fontsize=14)
plt.ylabel('Average Rating', fontsize=14)
plt.title('Average Difficulty vs Average Rating (Cleaned Data)', fontsize=16)
plt.grid(axis='both', linestyle='--', alpha=0.7)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()


q3r, q3p1 = pearsonr(avgDifficultyArray_clean, avgRatingQ3Array_clean)
q3rho, q3p2 = spearmanr(avgDifficultyArray_clean, avgRatingQ3Array_clean)
print(q3r)
print(q3rho)



avgDifficultyArray_clean_reshaped = avgDifficultyArray_clean.reshape(-1, 1)

# Create and fit the model
reg_model = LinearRegression()
reg_model.fit(avgDifficultyArray_clean_reshaped, avgRatingQ3Array_clean)

# Predict the ratings based on the difficulty values
q3predicted_ratings = reg_model.predict(avgDifficultyArray_clean_reshaped)

# Calculate residuals
q3residuals = avgRatingQ3Array_clean - q3predicted_ratings

# Plot the residuals in a histogram with a normal distribution line to show normality
plt.figure(figsize=(10, 6))

# Plot histogram of residuals
plt.hist(q3residuals, bins=30, density=True, alpha=0.6, color='purple', edgecolor='black')

# Plot the normal distribution fit line
mean_residuals = np.mean(q3residuals)
std_residuals = np.std(q3residuals)
x = np.linspace(min(q3residuals), max(q3residuals), 100)
plt.plot(x, stats.norm.pdf(x, mean_residuals, std_residuals), color='red', linewidth=2, label='Normal Distribution')

# Adding labels, title, and legend
plt.xlabel('Residuals')
plt.ylabel('Density')
plt.title('Histogram of Residuals with Normal Distribution Line')
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()



# %% Question 4


q4df = q1Df.dropna(subset=['numRatingsOnline'])

# Split the filtered dataset into two groups based on the percentage of online ratings
onlineProfessors = q4df[q4df['numRatingsOnline'] >= (0.5 * q4df['numRatings'])]
offlineProfessors = q4df[q4df['numRatingsOnline'] == 0]


online_ratings = onlineProfessors['AvgRating']
offline_ratings = offlineProfessors['AvgRating']


avgRatingBins = np.arange(0, 5.5, 0.5)  # Creating bins from 0 to 5 with a step of 0.5

# Plot histogram for online professors
plt.figure(figsize=(10, 6))
plt.hist(online_ratings, bins=avgRatingBins, alpha=0.5, label='Online Professors', color='blue', edgecolor='black')

# Plot histogram for offline professors
plt.hist(offline_ratings, bins=avgRatingBins, alpha=0.5, label='Offline Professors', color='yellow', edgecolor='black')

# Adding labels, title, and legend
plt.xlabel('Average Rating')
plt.ylabel('Frequency')
plt.title('Distribution of Average Ratings: Online vs Offline Professors')
plt.xticks(avgRatingBins)
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()


q4u, q4p = stats.mannwhitneyu(online_ratings, offline_ratings)
print(q4p)

q4onlineProfsMean = online_ratings.mean()
q4offlineProfsMean = offline_ratings.mean()
q4std_combined = np.sqrt(((online_ratings.std() ** 2) + (offline_ratings.std() ** 2)) / 2)
q4cohens_d = (q4onlineProfsMean - q4offlineProfsMean) / q4std_combined
print(q4cohens_d)


online_median = onlineProfessors['AvgRating'].median()
offline_median = offlineProfessors['AvgRating'].median()


# Create a new DataFrame for boxplot analysis
boxplot_data_q4 = q4df[['AvgRating', 'numRatingsOnline', 'numRatings']].copy()

# Label each professor as either 'Online' or 'Offline'
boxplot_data_q4['RatingSource'] = boxplot_data_q4.apply(
    lambda x: 'Online' if x['numRatingsOnline'] >= (0.5 * x['numRatings']) else 'Offline', axis=1
)

# Plot the box plot to compare average ratings between online and offline professors
plt.figure(figsize=(10, 6))
sns.boxplot(x='RatingSource', y='AvgRating', data=boxplot_data_q4, palette='Set3', width=0.5)

# Adding labels, title, and grid for better readability
plt.title('Box Plot of Average Ratings: Online vs Offline Professors')
plt.xlabel('Rating Source')
plt.ylabel('Average Rating')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# %% Question 5


noNanPropTakeAgainDf = numericalDf.dropna(subset=['propTakeAgain'])
nonNanPropWithThreshold = noNanPropTakeAgainDf[noNanPropTakeAgainDf["numRatings"] > 3]



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

q5r, q5p2 = pearsonr(avgRatingQ5Array_clean, propTakeAgainArray_clean)

print(q5r)


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

# Box Plot for Hot vs Not Hot Professors' Ratings

# Prepare the data for the boxplot
boxplot_data_q6 = pd.DataFrame({
    'Average Rating': pd.concat([hotProfRatings, notHotProfRatings]),
    'Category': ['Hot Professors'] * len(hotProfRatings) + ['Not Hot Professors'] * len(notHotProfRatings)
})

# Create the box plot
plt.figure(figsize=(10, 6))
sns.boxplot(x='Category', y='Average Rating', data=boxplot_data_q6, palette='Set3', width=0.5)
plt.title('Box Plot of Average Ratings: Hot vs Not Hot Professors')
plt.xlabel('Professor Category')
plt.ylabel('Average Rating')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

hot_median = hotProfs['AvgRating'].median()
notHot_median = notHotProfs['AvgRating'].median()

print(hot_median)
print(notHot_median)

# %% Question 7


q7df = noNanNumericalDf[noNanNumericalDf['numRatings'] > 3]


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

q8corr_matrix = q8Df.corr()


plt.figure(figsize=(12, 10))
sns.heatmap(q8corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix for Features')
plt.show()

# Train-test split
q8X_train, q8X_test, q8y_train, q8y_test = train_test_split(q8X, q8y, test_size=0.2, random_state=SEED)

# Set a list of alpha values to test
alphas = np.logspace(-4, 4, 50)

# RidgeCV will internally do cross-validation for you
ridge_cv = RidgeCV(alphas=alphas, cv=5, scoring='r2')  # You can adjust the number of folds
ridge_cv.fit(q8X_train, q8y_train)

# Print the best alpha
print("Best alpha (RidgeCV):", ridge_cv.alpha_)

# Initialize and fit the regression model
q8reg_model = Ridge(alpha=11.5)
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





# %% Question 9

q9X = q1Df[['AvgRating']].values  
q9y = q1Df['Hot'].values       
   
# Split the data into training and testing sets
q9X_train, q9X_test, q9y_train, q9y_test = train_test_split(q9X, q9y, test_size=0.2, random_state=SEED)

# Train Logistic Regression model
q9model = LogisticRegression()
q9model.fit(q9X_train, q9y_train)

# Make predictions with probabilities on the test set
q9y_pred_prob = q9model.predict_proba(q9X_test)[:, 1]  # Probability of pepper being 1

# Calculate FPR, TPR, and thresholds for the ROC curve
q9_fpr, q9_tpr, q9_thresholds = roc_curve(q9y_test, q9y_pred_prob)

# Find the optimal threshold
optimal_idx = np.argmax(q9_tpr - q9_fpr)
optimal_threshold = q9_thresholds[optimal_idx]
print(f"Optimal Threshold: {optimal_threshold:.3f}")

# Apply the optimal threshold to make predictions
q9y_pred_custom = (q9y_pred_prob >= optimal_threshold).astype(int)

# Calculate the AUC score with the predicted probabilities
q9_auc = roc_auc_score(q9y_test, q9y_pred_prob)
print(f"AUC: {q9_auc:.3f}")

# Calculate accuracy using the custom threshold
q9_accuracy = accuracy_score(q9y_test, q9y_pred_custom)
print(f"Accuracy with Optimal Threshold: {q9_accuracy:.3f}")

# Plot Confusion Matrix using the custom threshold
q9_conf_matrix = confusion_matrix(q9y_test, q9y_pred_custom)
plt.figure(figsize=(8, 6))
sns.heatmap(q9_conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix with Optimal Threshold')
plt.show()

# Plot ROC Curve and highlight the optimal point
plt.figure(figsize=(10, 6))
plt.plot(q9_fpr, q9_tpr, color='blue', linewidth=2, label='ROC Curve')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # Diagonal line for random guess
plt.scatter(q9_fpr[optimal_idx], q9_tpr[optimal_idx], color='red', label=f'Optimal Threshold = {optimal_threshold:.3f}', zorder=5)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve with Optimal Threshold')
plt.legend()
plt.show()

q9hot_count = q9y.sum()  # Count of hot professors (Hot = 1)
q9not_hot_count = len(q9y) - q9hot_count  # Count of not hot professors (Hot = 0)

print(q9hot_count)
print(q9not_hot_count)


# %% Question 10


q10X = nonNanPropWithThreshold.drop(columns=['Hot'])  
q10y = nonNanPropWithThreshold['Hot']  


# Split the data into training and testing sets
q10X_train, q10X_test, q10y_train, q10y_test = train_test_split(q10X, q10y, test_size=0.2, random_state=SEED)

# Train Logistic Regression model
q10model = LogisticRegression(max_iter=1000)
q10model.fit(q10X_train, q10y_train)

# Make predictions with probabilities on the test set
q10y_pred_prob = q10model.predict_proba(q10X_test)[:, 1]  # Probability of pepper being 1

# Calculate FPR, TPR, and thresholds for the ROC curve
q10_fpr, q10_tpr, q10_thresholds = roc_curve(q10y_test, q10y_pred_prob)

# Find the optimal threshold
optimal_idx = np.argmax(q10_tpr - q10_fpr)
optimal_threshold = q10_thresholds[optimal_idx]
print(f"Optimal Threshold: {optimal_threshold:.3f}")

# Apply the optimal threshold to make predictions
q10y_pred_custom = (q10y_pred_prob >= optimal_threshold).astype(int)

# Calculate the AUC score with the predicted probabilities
q10_auc = roc_auc_score(q10y_test, q10y_pred_prob)
print(f"AUC: {q10_auc:.3f}")

# Calculate accuracy using the custom threshold
q10_accuracy = accuracy_score(q10y_test, q10y_pred_custom)
print(f"Accuracy with Optimal Threshold: {q10_accuracy:.3f}")

# Plot Confusion Matrix using the custom threshold
q10_conf_matrix = confusion_matrix(q10y_test, q10y_pred_custom)
plt.figure(figsize=(8, 6))
sns.heatmap(q10_conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix with Optimal Threshold')
plt.show()

# Plot ROC Curve and highlight the optimal point
plt.figure(figsize=(10, 6))
plt.plot(q10_fpr, q10_tpr, color='blue', linewidth=2, label='ROC Curve')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # Diagonal line for random guess
plt.scatter(q10_fpr[optimal_idx], q10_tpr[optimal_idx], color='red', label=f'Optimal Threshold = {optimal_threshold:.3f}', zorder=5)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve with Optimal Threshold')
plt.legend()
plt.show()

q10hot_count = q10y.sum()  # Count of hot professors (Hot = 1)
q10not_hot_count = len(q10y) - q10hot_count  # Count of not hot professors (Hot = 0)




# %% Extra credit

# Merge numerical and qualitative datasets for state and major analysis
mergedDf = pd.concat([numericalDf, qualitativeDf], axis=1)

# Rename columns for clarity
mergedDf.columns = ["AvgRating", "AvgDifficulty", "numRatings", "Hot", "propTakeAgain", 
                     "numRatingsOnline", "Male", "Female", "Major", "University", "State"]


mergedDf = mergedDf.dropna(subset=['AvgRating'])

ThresholdergedDf = mergedDf[mergedDf['numRatings'] > 4]

# Find unique majors and states in the original dataset
original_majors = set(mergedDf['Major'].unique())
original_states = set(mergedDf['State'].unique())

# Find unique majors and states in the filtered dataset
filtered_majors = set(ThresholdergedDf['Major'].unique())
filtered_states = set(ThresholdergedDf['State'].unique())

# Determine eliminated majors and states
eliminated_majors = original_majors - filtered_majors
eliminated_states = original_states - filtered_states

# Define a set of all US state abbreviations
us_states = {
    'AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA', 'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 
    'LA', 'ME', 'MD', 'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ', 'NM', 'NY', 'NC', 'ND', 
    'OH', 'OK', 'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY'
}

# Add a new column to classify professors as US or Non-US based on the state
ThresholdergedDf['Location'] = ThresholdergedDf['State'].apply(lambda x: 'US' if x in us_states else 'Non-US')

# Group by Location and calculate summary statistics
location_summary = ThresholdergedDf.groupby('Location').agg(
    avg_rating=('AvgRating', 'mean'),
    avg_difficulty=('AvgDifficulty', 'mean'),
    num_professors=('AvgRating', 'count'),
    total_ratings=('numRatings', 'sum'),
    online_ratings=('numRatingsOnline', 'sum')
).reset_index()

# Print the summary table
print(location_summary)

# Plot histogram for US vs Non-US Professors
plt.figure(figsize=(10, 6))
bins = np.arange(0, 5.5, 0.5)  # Bins from 0 to 5 with a step of 0.5

# Histogram for US Professors
plt.hist(ThresholdergedDf[ThresholdergedDf['Location'] == 'US']['AvgRating'], bins=bins, alpha=0.5, 
         label='US Professors', color='blue', edgecolor='black')

# Histogram for Non-US Professors
plt.hist(ThresholdergedDf[ThresholdergedDf['Location'] == 'Non-US']['AvgRating'], bins=bins, alpha=0.5, 
         label='Non-US Professors', color='yellow', edgecolor='black')

# Adding labels, title, and legend
plt.title('Distribution of Average Ratings: US vs Non-US Professors')
plt.xlabel('Average Rating')
plt.ylabel('Frequency')
plt.xticks(bins)
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# Boxplot for US vs Non-US Professors
plt.figure(figsize=(10, 6))
sns.boxplot(x='Location', y='AvgRating', data=ThresholdergedDf, palette='Set3', width=0.5)

# Adding labels, title, and grid
plt.title('Box Plot of Average Ratings: US vs Non-US Professors')
plt.xlabel('Location')
plt.ylabel('Average Rating')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# Separate the ratings for US and Non-US professors
us_ratings = ThresholdergedDf[ThresholdergedDf['Location'] == 'US']['AvgRating']
non_us_ratings = ThresholdergedDf[ThresholdergedDf['Location'] == 'Non-US']['AvgRating']

# Perform Mann-Whitney U-Test for US vs Non-US professors
u_statistic_us, p_value_us = stats.mannwhitneyu(us_ratings, non_us_ratings, alternative='two-sided')


# Calculate Cohen's d for US vs Non-US
mean_us = us_ratings.mean()
mean_non_us = non_us_ratings.mean()
std_combined = np.sqrt(((us_ratings.std() ** 2) + (non_us_ratings.std() ** 2)) / 2)
cohens_d_us = (mean_us - mean_non_us) / std_combined



# Define STEM-related keywords
stem_keywords = ["Engineering", "Science", "Mathematics", "Technology", "STEM", "Physics", "Chemistry", "Biology"]

# Add a new column to classify professors as STEM or Non-STEM
ThresholdergedDf['STEM'] = ThresholdergedDf['Major'].str.contains('|'.join(stem_keywords), case=False, na=False)

# Calculate summary statistics for STEM and Non-STEM professors
stem_summary = ThresholdergedDf.groupby('STEM').agg(
    median_rating=('AvgRating', 'median'),
    median_difficulty=('AvgDifficulty', 'median'),
    num_professors=('AvgRating', 'count'),
    total_ratings=('numRatings', 'sum')
).reset_index()

# Rename categories for clarity
stem_summary['STEM'] = stem_summary['STEM'].replace({True: 'STEM', False: 'Non-STEM'})

# Print summary
print(stem_summary)

# Visualization: Histogram of Ratings for STEM vs. Non-STEM
plt.figure(figsize=(10, 6))
bins = np.arange(0, 5.5, 0.5)  # Bins from 0 to 5 with a step of 0.5

# Histogram for STEM professors
plt.hist(ThresholdergedDf[ThresholdergedDf['STEM'] == True]['AvgRating'], bins=bins, alpha=0.5, 
         label='STEM Professors', color='blue', edgecolor='black')

# Histogram for Non-STEM professors
plt.hist(ThresholdergedDf[ThresholdergedDf['STEM'] == False]['AvgRating'], bins=bins, alpha=0.5, 
         label='Non-STEM Professors', color='yellow', edgecolor='black')

# Adding labels, title, and legend
plt.title('Distribution of Average Ratings: STEM vs Non-STEM Professors')
plt.xlabel('Average Rating')
plt.ylabel('Frequency')
plt.xticks(bins)
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# Boxplot for STEM vs Non-STEM Professors
plt.figure(figsize=(10, 6))
sns.boxplot(x='STEM', y='AvgRating', data=ThresholdergedDf, palette='Set3', width=0.5)

# Adding labels, title, and grid
plt.title('Box Plot of Average Ratings: STEM vs Non-STEM Professors')
plt.xlabel('Category')
plt.ylabel('Average Rating')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# Perform Mann-Whitney U-Test for STEM vs Non-STEM
stem_ratings = ThresholdergedDf[ThresholdergedDf['STEM'] == True]['AvgRating']
non_stem_ratings = ThresholdergedDf[ThresholdergedDf['STEM'] == False]['AvgRating']

# Perform Mann-Whitney U-Test for STEM vs Non-STEM professors
u_statistic_stem, p_value_stem = stats.mannwhitneyu(stem_ratings, non_stem_ratings, alternative='two-sided')
print(f"Mann-Whitney U Test (STEM vs Non-STEM): U-statistic = {u_statistic_stem}, p-value = {p_value_stem}")

# Calculate Cohen's d for STEM vs Non-STEM
median_stem = stem_ratings.median()
median_non_stem = non_stem_ratings.median()
std_combined_stem = np.sqrt(((stem_ratings.std() ** 2) + (non_stem_ratings.std() ** 2)) / 2)
cohens_d_stem = (median_stem - median_non_stem) / std_combined_stem
print(f"Cohen's d (STEM vs Non-STEM): {cohens_d_stem}")
