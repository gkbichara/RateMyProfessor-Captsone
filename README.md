# Capstone Project: Professor Ratings Analysis

## Overview

This project aims to analyze professor ratings from a dataset that includes metrics such as average ratings, difficulty levels, the number of ratings, "hot" indicators, and student willingness to retake a professor's class. The analysis is performed using Python and various statistical techniques to better understand factors that influence professor ratings and student satisfaction.

The analysis includes:
- Exploratory Data Analysis (EDA) to visualize distributions.
- Comparison of average ratings between different groups (e.g., "Hot" vs "Not Hot" professors).
- Statistical tests (e.g., Mann-Whitney U and correlation coefficients) to understand relationships between variables.

## Project Structure

- **`Capstone.py`**: Main script that contains the data loading, analysis, and visualization logic.
- **Datasets**: The project uses two CSV files for analysis:
  - `rmpCapstoneNum.csv`: Contains numerical data such as average ratings, difficulty, and number of ratings.
  - `rmpCapstoneQual.csv`: Contains qualitative data like major, university, and state.
- **`README.md`**: Project description and instructions (this file).

## Getting Started

### Prerequisites

To run this project, you'll need:
- **Python 3.7+**
- The following Python libraries:
  - `numpy`
  - `pandas`
  - `matplotlib`
  - `scipy`
  - `sklearn`

You can install the required packages using pip:
```sh
pip install numpy pandas matplotlib scipy sklearn
```

### Running the Project

1. **Clone the repository**:
   ```sh
   git clone https://github.com/username/capstone-project.git
   cd capstone-project
   ```

2. **Run the main script**:
   ```sh
   python Capstone.py
   ```

### Data

The numerical dataset (`rmpCapstoneNum.csv`) includes the following columns:
- `AvgRating`: The average rating given to a professor.
- `AvgDifficulty`: The average difficulty rating.
- `numRatings`: The total number of ratings received.
- `Hot`: Binary indicator of whether the professor is considered "hot".
- `propTakeAgain`: Proportion of students willing to retake the professor's class.
- `numRatingsOnline`, `Male`, `Female`: Additional attributes related to the professors.

The qualitative dataset (`rmpCapstoneQual.csv`) includes:
- `Major/Field`: The field of study or subject area.
- `University`, `State`: Location-based attributes of professors.

## Features and Analysis

- **EDA**: Visualizations of distributions of ratings, difficulty, and other factors using histograms.
- **Group Comparisons**: Comparisons of ratings between different groups (e.g., "Hot" vs "Not Hot" professors, male vs female professors).
- **Statistical Tests**: Use of Mann-Whitney U test to determine statistically significant differences between groups.
- **Correlation Analysis**: Pearson's r and Spearman's rho calculations to measure relationships between metrics like average ratings and difficulty or student willingness to retake the course.

## Results

The project provides insights into:
- Factors that influence professor ratings.
- Statistical differences between various groups of professors (e.g., gender, popularity).
- Correlations between ratings, difficulty, and the likelihood of students retaking a course.

## Future Work

- **Feature Engineering**: Creating additional features from the existing dataset to improve the analysis.
- **Machine Learning**: Develop a predictive model to estimate professor ratings based on their attributes.
- **Further Visualizations**: Adding interactive visualizations to better explore relationships between different factors.

