# Assessing Professor Effectiveness (APE)

**Galal Bichara**  
**Principles of Data Science – DS UA 112**  
**Instructor: Pascal Wallisch**  

This repository contains the materials for the capstone project titled *Assessing Professor Effectiveness (APE)*. The goal of this project is to analyze professor ratings sourced from RateMyProfessors.com (RMP) to answer key questions about factors influencing professor evaluations, uncover insights, and contribute to a deeper understanding of the data science workflow.

---

## Project Description

RateMyProfessors.com provides a public dataset of professor ratings contributed by students. Although subject to potential biases and limitations, this dataset offers a unique opportunity to analyze trends in professor evaluations. Using Python and statistical techniques, this project explores the relationships between factors such as gender, teaching experience, class difficulty, and modality on professor ratings. 

---

## Dataset Information

The project uses two datasets:

1. **rmpCapstoneNum.csv**  
   Contains numerical data with 89,893 rows and the following columns:
   - `AvgRating` - Average Rating
   - `AvgDifficulty` - Average Difficulty
   - `numRatings` - Number of Ratings
   - `Hot` - Received a “pepper” (Hotness indicator)
   - `propTakeAgain` - Proportion of students willing to take the class again
   - `numRatingsOnline` - Number of ratings from online classes
   - `Male` - Male indicator (Boolean)
   - `Female` - Female indicator (Boolean)

2. **rmpCapstoneQual.csv**  
   Contains qualitative data with 89,893 rows and the following columns:
   - `Major` - Major/Field
   - `University` - University
   - `State` - State (US and international abbreviations)

---

## Questions Explored in the Project

1. **Is there evidence of a pro-male gender bias in professor evaluations?**  
   Explores whether male professors have higher ratings compared to female professors using statistical significance tests.

2. **Does teaching experience impact the quality of teaching?**  
   Investigates the effect of experience (proxied by the number of ratings) on professor ratings.

3. **What is the relationship between average rating and average difficulty?**  
   Analyzes whether more difficult professors receive lower ratings and calculates correlation coefficients.

4. **How do ratings differ between professors teaching online and offline classes?**  
   Compares ratings of professors who teach primarily online versus offline using a custom data split.

5. **What is the relationship between the proportion of students willing to retake a class and the professor's rating?**  
   Examines the strength of the correlation between these variables.

6. **Do professors marked as "hot" receive higher ratings?**  
   Tests whether professors with the "pepper" indicator have significantly different ratings than others.

7. **Can we predict ratings based solely on difficulty?**  
   Develops a regression model and evaluates its performance using metrics like R² and RMSE.

8. **Can we predict ratings using all available factors?**  
   Builds a multivariate regression model and discusses individual predictors and model performance.

9. **Can a professor’s "hotness" be predicted from average ratings?**  
   Builds a classification model using logistic regression and evaluates its performance using metrics like AUC.

10. **Can "hotness" be predicted from all available factors?**  
    Builds a comprehensive classification model and compares its performance with the single-factor model.

11. **Extra Credit:**
    - Compares US vs. non-US professors.
    - Compares STEM vs. non-STEM professors.
    - Identifies eliminated states and majors during preprocessing.

---

## Project Files

- `Capstone.py`: Python script containing the code for data cleaning, analysis, visualizations, and modeling.  
- `rmpCapstoneNum.csv`: Dataset with numerical information about professors.  
- `rmpCapstoneQual.csv`: Dataset with qualitative information about professors.  
- `Capstone project spec sheet.pdf`: Detailed project specifications and guidelines.  
- `Galal Bichara - PODS - Capstone.pdf`: Final report containing answers, visualizations, and statistical analyses.

---

## Key Findings

- **Gender Bias:** Male professors have slightly higher ratings than female professors, but the effect size is minimal.  
- **Experience:** More experienced professors generally receive higher ratings.  
- **Difficulty vs. Rating:** Higher difficulty is associated with lower ratings, as shown by a negative correlation.  
- **Modality:** Offline professors receive higher ratings compared to online professors.  
- **Take-Again Proportion:** A strong positive correlation exists between ratings and the proportion of students willing to retake a class.  
- **"Hotness" and Ratings:** Professors marked as "hot" tend to have significantly higher ratings.  

Additional insights are documented in the final report.

---

## How to Use This Repository

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/ape-capstone.git
   cd ape-capstone
