# NBA Game Prediction Model

## Overview

This project explores how machine learning can be used to predict **NBA game outcomes** using historical performance data.

Using NBA game data from the **2019–2024 seasons (~10,000 games)**, I built an end-to-end machine learning pipeline that processes game data, engineers predictive features, trains models, and generates predictions for upcoming games.

The current model achieved **67.8% accuracy predicting NBA game winners**, which is considered strong for sports prediction problems due to the naturally unpredictable nature of game outcomes.

The motivation behind this project was to apply my knowledge of **Python and machine learning** to a real-world dataset while building a complete system similar to what a machine learning engineer might develop in practice. As someone who has been passionate about basketball for many years and played growing up, this project was also a fun way to combine sports with data science.

---

# What This Project Does

This system:

• Collects historical NBA game data  
• Engineers predictive features from past games  
• Trains a machine learning model to predict game winners  
• Uses the trained model to generate predictions for upcoming games  
• Compares model probabilities with Vegas implied probabilities  

The project is structured to make it easy to experiment with new features, models, and prediction methods.

---

# Machine Learning Approach

The model predicts whether the **home team will win a game** using historical performance data.

Feature engineering includes:

• **Team Elo ratings** to estimate team strength over time  
• **Rolling performance metrics** (points, rebounds, assists over recent games)  
• **Offensive and defensive efficiency ratings**  
• **Rest advantage and schedule difficulty**  
• **Home court performance trends**  
• **Matchup-based feature differences between teams**

To simulate real prediction conditions, the model uses **time-aware feature engineering** and a **chronological train/test split**, ensuring that predictions only use information available before each game.

---

# Model Performance

Dataset: **NBA games from 2019–2024 seasons (~10,000 games)**

Model: **Random Forest Classifier**

Performance:

• **67.8% accuracy predicting NBA game winners**

Because sports outcomes are inherently volatile, models that achieve **60–65% accuracy** are generally considered strong in sports analytics. The current model performs above that benchmark.

---

# Technologies Used

- Python  
- Pandas  
- NumPy  
- Scikit-Learn  
- Random Forest
- NBA API Integration
- Joblib
- Plans to test XGBoost and ensemble methods

---

**train_model.py**  
Builds the feature dataset, trains the machine learning model, evaluates accuracy, and saves the trained model.

**predict_games.py**  
Loads the trained model and generates predictions for today's NBA games using the most recent team statistics.

---

# How To Run The Project

### 1. Clone the repo and cd into the root folder (from terminal) 
git clone https://github.com/BryllieBrinkley/NBA-Game-Prediction-Model.git

cd NBA-Game-Prediction-Model

--

### 2. Install required Python libraries
pip install pandas numpy scikit-learn xgboost nba_api requests joblib

---

### 3. Run predictions for today's games

python src/predict_games.py

Example output:
Warriors vs Suns | Model: 0.52 | Vegas: 0.48 | Spread: +2.5 | Edge: +0.04 | Prediction: Warriors

---

### 4. Train the model yourself (optional)

If you want to retrain the model using the included dataset:
python src/train_model.py


This will:

• rebuild the feature dataset  
• train the model  
• evaluate performance  
• save the trained model to `/models/winner_model.pkl`

---

# Skills Demonstrated

Machine Learning  
Feature Engineering  
Predictive Modeling  
Python Development  
Data Processing Pipelines  
API Integration  
Statistical Modeling  
Sports Analytics  

---

# Future Improvements
Future improvements to the project include:

• Expanding the model to predict **game totals (over/under)**  
• Testing additional models such as **XGBoost and ensemble approaches**  
• Adding **automated backtesting against betting market odds**  
• Building a **dashboard for visualizing predictions**

---

# Author

**Jibryll Brinkley**
