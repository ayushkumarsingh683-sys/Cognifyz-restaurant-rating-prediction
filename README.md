# Cognifyz-restaurant-rating-prediction

A Machine Learning project built using Python and `scikit-learn` to predict the aggregate rating of restaurants based on features like user engagement, cost, and cuisine types. 

## 📌 Project Objective
The main goal of this project is to build an end-to-end machine learning regression pipeline that predicts a restaurant's **Aggregate Rating**. The workflow involves:
1. Handling missing values and data cleaning.
2. Encoding categorical attributes into numerical formats.
3. Training and evaluating a regression model.
4. Analyzing feature importances to determine what drives restaurant ratings.

---

## 📊 Dataset Features
The model utilizes a restaurant dataset (`Dataset.csv`) with the following key attributes:
* **Votes (Numerical):** The total number of reviews/votes a restaurant has received.
* **Average Cost for two (Numerical):** Expected cost for two diners.
* **Has Table booking (Categorical):** Whether the restaurant accepts reservations (`Yes`/`No`).
* **Cuisines (Categorical):** The types of food served (e.g., North Indian, Chinese, Italian, Fast Food).
* **Aggregate rating (Target Variable):** The restaurant's overall score out of 5.

---

## 🛠️ Tech Stack & Dependencies
This project requires Python 3.x along with the following libraries:
* **Pandas** - Data manipulation and loading
* **NumPy** - Numerical operations
* **Scikit-Learn** - Data preprocessing pipelines, model selection, and machine learning
* **Matplotlib** - Feature importance visualization

To install dependencies, run:
```bash
pip install numpy pandas scikit-learn matplotlib
⚙️ Methodology & Pipeline Implementation
1. Data Preprocessing
Missing Values: Numerical features are imputed using the column median. Categorical features are imputed using the most frequent value to ensure no data is dropped.

Categorical Encoding: Text features (Cuisines and Has Table booking) are converted into binary columns using One-Hot Encoding.

Data Splitting: The processed data is split into an 80% training set and a 20% testing set to robustly evaluate performance.

2. Model Choice
A Random Forest Regressor (n_estimators=100) was chosen for this task. Random Forests excel at capturing complex, non-linear relationships between mix-typed features (like cost and cuisine type) without requiring intensive manual feature scaling.

📈 Key Insights & Results
Model Performance Metrics
The model's accuracy on unseen testing data is captured via standard regression evaluation metrics:

Mean Squared Error (MSE)

Root Mean Squared Error (RMSE)

R-squared (R 
2
 ) Score

Feature Importance Analysis
Upon plotting the model's inner weights, the project reveals a clear hierarchy in what influences restaurant ratings:

Votes (Highest Pull): By an overwhelming margin, customer popularity/review volume (Votes) is the strongest predictor of a higher aggregate rating.

Average Cost for two: Acts as a secondary factor, reflecting how pricing impacts consumer expectations and rating behaviors.

Cuisine & Bookings: Specific cuisine configurations hold a minor localized impact relative to overall user engagement.

🚀 How to Run the Project
Clone this repository to your local machine.

Place your Dataset.csv file inside the root folder.

Run the python script:

Bash
python restaurantraiting.py
A window will pop up showing the Top Features Influencing Restaurant Ratings chart, and performance metrics will print directly to your terminal.
