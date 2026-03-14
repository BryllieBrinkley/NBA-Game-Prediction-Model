# NBA Game Prediction Model

## Overview
This project is a production-ready machine learning pipeline for predicting NBA game outcomes. It leverages historical and live NBA data, advanced feature engineering, and state-of-the-art machine learning models to forecast game winners and (soon) point totals. The modular, extensible design makes it easy to experiment with new features, algorithms, and prediction tasks. This project demonstrates real-world data science, MLOps, and sports analytics skills using modern Python tools.

## Features
- Predicts NBA game winners using ensemble ML models (Random Forest, XGBoost, etc.)
- Fetches and processes live betting odds
- Integrates multi-season NBA data
- Modular codebase for easy extension (add new models, features, or prediction tasks)
- Ready for point total prediction and deep learning extensions
- Automated data collection, preprocessing, training, and prediction workflows

## Project Structure
- **src/**: Core source code (data processing, model training, prediction)
    - `get_games.py`: Fetches NBA game data and odds
    - `process_games.py`: Cleans and prepares datasets
    - `train_model.py`: Trains machine learning models
    - `predict_games.py`: Runs predictions on upcoming games
- **models/**: Trained model files (e.g., `winner_model.pkl`)
- **data/**: Raw and processed datasets (CSV files)

## Setup Instructions
1. **Python Version**: 3.8 or higher
2. **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
    _Key dependencies:_
    - scikit-learn
    - xgboost
    - pandas
    - nba_api
    - (Optional for future work: pytorch, tensorflow, lightgbm)
3. **Environment Variables**:
    - Set API keys (e.g., `ODDS_API_KEY`) in your shell or a `.env` file.

## Usage Examples
**Train a Model:**
```bash
python src/train_model.py
```
**Predict Upcoming Games:**
```bash
python src/predict_games.py
```
**Fetch and Process New Data:**
```bash
python src/get_games.py
python src/process_games.py
```

## Sample Output
```
Today's Games Predictions

Lakers vs Celtics | Model: 0.67 | Vegas: 0.61 | Spread: -3.5 | Edge: +0.06 | Prediction: Lakers | Home injuries: 0.0 | Away injuries: 1.0 | BET HOME
Warriors vs Suns | Model: 0.48 | Vegas: 0.52 | Spread: +2.0 | Edge: -0.04 | Prediction: Suns | Home injuries: 0.5 | Away injuries: 0.0 | NO BET
```

## Extending the Project
- **Point Total Prediction**: Add regression models and features for predicting total points (see `evaluate_totals_model` in `train_model.py`).
- **New ML Frameworks**: Integrate PyTorch, TensorFlow, or LightGBM for advanced modeling.
- **Custom Features**: Add new feature engineering steps in `src/` scripts.
- **Web API or Dashboard**: Deploy predictions as a REST API (Flask/FastAPI) or interactive dashboard (Streamlit, Dash).
- **CI/CD**: Add GitHub Actions for automated testing and deployment.

## Future Work / Roadmap
- Point total prediction and betting signals
- Deep learning models (PyTorch, TensorFlow)
- Automated hyperparameter tuning
- Web dashboard for live predictions
- Continuous integration and deployment
- Model and data versioning (DVC, MLflow)

## Acknowledgements
- **Data Sources**: [nba_api](https://github.com/swar/nba_api), public betting odds
- **Libraries/Frameworks**: scikit-learn, xgboost, pandas, nba_api, pytorch, tensorflow, lightgbm

## Contact & Contributions
For questions or contributions, open an issue or submit a pull request.

Project maintained by [YOUR NAME OR GITHUB HANDLE]
