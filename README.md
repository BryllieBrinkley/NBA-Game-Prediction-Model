# NBA Game Prediction Model
Personal Machine Learning Project

This project uses machine learning to predict NBA game winners and point totals using historical NBA data from the 2019–2024 seasons.

The model was trained on around 10,000 past games and achieved 67.8% accuracy predicting game winners.

My original motivation for this project was to build my own machine learning model capable of predicting NBA game outcomes for sports betting. I wanted to challenge existing prediction models and see if I could develop one that performs competitively using historical NBA data. Through this process I have built an end-to-end pipeline, including collecting data, engineering meaningful features, training models, and evaluating predictions. This project was especially meaningful to me because basketball is a sport I have been passionate about since my childhood, and I was very pleased to see my model get the first 4 picks of the day correctly the very next day. 

# What This Project Does
This project:
• Collects historical NBA game data  
• Creates useful features from that data to train the ML model.
• Trains a machine learning model  
• Uses that model to predict future games
- Plan to expand project with new features below

# Skills Demonstrated
Skills Used:
• Machine Learning  
• Feature Engineering
• Data Processing Pipelines
• API Integration  
• Data Analytics
• Model Evaluation  
• Python Development

Technologies used(*working):
- nba api
- Python  
- Pandas  
- NumPy  
- Scikit-Learn
- XGBoost (for baseline model)

# Model Performance:
Training data - NBA games from 2019–2024 seasons
Model performance - Model achieved a 67.8% accuracy predicting NBA game winners. Due to the unpredicatable nature of sporting events, models that achieve around a 60–65% accuracy are considered strong.

# To run the project yourself, follow these steps.

### 1 Clone the repository and cd into the root folder
git clone https://github.com/YOUR_USERNAME/NBA-Game-Prediction-Model.git
cd NBA-Game-Prediction-Model

### 2. Install the required libraries (used for loading data, training the model, generating predictions)
pip3 install pandas numpy scikit-learn xgboost nba_api requests joblib

### 3. Run the Prediction script 
python3 src/predict_games.py

### 4. (Optional) Train Model Yourself
If you'd like to retrain the model using the historical data included in the repository:
python3 src/train_model.py

### 5. Example Output

(python3 predict_games.py)
Lakers vs Nuggets | Model: 0.49 | Vegas: 0.44 | Spread: +2.5 | Edge: +0.05 | Prediction: Nuggets | Home injuries: 0.0 | Away injuries: 0.0 | BET HOME
Spurs vs Hornets | Model: 0.72 | Vegas: 0.68 | Spread: -6.5 | Edge: +0.05 | Prediction: Spurs | Home injuries: 0.0 | Away injuries: 0.0 | NO BET
Warriors vs Timberwolves | Model: 0.52 | Vegas: 0.32 | Spread: +6.0 | Edge: +0.20 | Prediction: Warriors | Home injuries: 0.0 | Away injuries: 0.0 | BET HOME
Celtics vs Wizards | Model: 0.83 | Vegas: 0.92 | Spread: -19.5 | Edge: -0.09 | Prediction: Celtics | Home injuries: 0.0 | Away injuries: 0.0 | BET AWAY

Author
Jibryll Brinkley
