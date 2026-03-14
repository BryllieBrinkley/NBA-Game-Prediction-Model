
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    RandomForestRegressor,
    RandomForestClassifier,
    GradientBoostingClassifier,
    VotingClassifier,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor, XGBClassifier
import os
import joblib

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "nba_games_multi_season.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "winner_model.pkl")

N_SPLITS = 5
RANDOM_STATE = 42

#Trains a ML model to predict NBA game outcomes based on historical game data, team stats, rest days, and ELO ratings
# Pipeline:
# 1. Load and clean historical NBA game data
# 2. Engineer predictive features (ELO, form, schedule, efficiency metrics)
# 3. Train classification models
# 4. Evaluate accuracy on a time-based holdout set
# 5. Save the trained model for use in live predictions

# The model is designed to mimic real betting conditions by only using
# information available *before* each game, making it suitable for predicting future games.

def load_data(path: str) -> pd.DataFrame:
# Load and clean NBA games data

    df = pd.read_csv(path).copy()
    df.columns = df.columns.str.strip()
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df = df.sort_values(["GAME_DATE", "GAME_ID"]).reset_index(drop=True)

    # Merges raw data with duplicate game entries into a single row per game
    home = df[df["MATCHUP"].str.contains("vs.", na=False)].copy()
    away = df[df["MATCHUP"].str.contains("@", na=False)].copy()
    home = home.add_suffix("_home")
    away = away.add_suffix("_away")
    games = home.merge(
        away,
        left_on="GAME_ID_home",
        right_on="GAME_ID_away",
        how="inner",
    )
    games["total_points"] = games["PTS_home"] + games["PTS_away"]
    games["home_win"] = (games["PTS_home"] > games["PTS_away"]).astype(int)
    games["margin_home"] = games["PTS_home"] - games["PTS_away"]
    games = games[
        (games["total_points"] > 150) & (games["total_points"] < 300)
    ].reset_index(drop=True)
    games = games.sort_values(["GAME_DATE_home", "GAME_ID_home"]).reset_index(drop=True)
    return games


def simple_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Estimate possessions for home and away teams using common formula
    df["poss_home"] = (
        df["FGA_home"] - df["OREB_home"] + df["TOV_home"] + 0.44 * df["FTA_home"]
    )
    df["poss_away"] = (
        df["FGA_away"] - df["OREB_away"] + df["TOV_away"] + 0.44 * df["FTA_away"]
    )
    # Offensive Rating = points scored per 100 possessions
    df["off_rating_home_game"] = (
        100 * df["PTS_home"] / df["poss_home"].replace(0, np.nan)
    )
    df["off_rating_away_game"] = (
        100 * df["PTS_away"] / df["poss_away"].replace(0, np.nan)
    )
    df["def_rating_home_game"] = (
        100 * df["PTS_away"] / df["poss_away"].replace(0, np.nan)
    )
    df["def_rating_away_game"] = (
        100 * df["PTS_home"] / df["poss_home"].replace(0, np.nan)
    )
    df["ts_home_game"] = df["PTS_home"] / (
        2 * (df["FGA_home"] + 0.44 * df["FTA_home"]).replace(0, np.nan)
    )
    df["ts_away_game"] = df["PTS_away"] / (
        2 * (df["FGA_away"] + 0.44 * df["FTA_away"]).replace(0, np.nan)
    )
    return df

# Rest and Schedule Features
def add_rest_schedule_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    home_last = df.groupby("TEAM_ID_home")["GAME_DATE_home"].shift(1)
    away_last = df.groupby("TEAM_ID_away")["GAME_DATE_home"].shift(1)
    df["home_rest_days"] = (df["GAME_DATE_home"] - home_last).dt.days
    df["away_rest_days"] = (df["GAME_DATE_home"] - away_last).dt.days
    df["rest_diff"] = df["home_rest_days"] - df["away_rest_days"]
    df["home_games_last7"] = df.groupby("TEAM_ID_home")["GAME_DATE_home"].transform(
        lambda x: x.shift(1).rolling(7).count()
    )
    df["away_games_last7"] = df.groupby("TEAM_ID_away")["GAME_DATE_home"].transform(
        lambda x: x.shift(1).rolling(7).count()
    )
    df["schedule_diff"] = df["home_games_last7"] - df["away_games_last7"]
    return df

# ELO Features:
# - Each NBA team starts with an ELO rating of 1500 and ter each game, ratings are updated based on outcome.
# dynamic estimate of team strength throughout the season.
def add_elo_features(df: pd.DataFrame, k: float = 20.0) -> pd.DataFrame:
    """Add ELO rating features to DataFrame."""
    df = df.copy()
    teams = pd.concat([df["TEAM_ID_home"], df["TEAM_ID_away"]]).unique()
    elo = {team: 1500.0 for team in teams}
    home_elos = []
    away_elos = []
    for _, row in df.iterrows():
        home = row["TEAM_ID_home"]
        away = row["TEAM_ID_away"]
        home_elo = elo[home]
        away_elo = elo[away]
        home_elos.append(home_elo)
        away_elos.append(away_elo)
        expected_home = 1 / (1 + 10 ** ((away_elo - home_elo) / 400))
        actual_home = row["home_win"]
        margin = abs(row["margin_home"])
        mov_multiplier = np.log(margin + 1)
        elo_change = k * mov_multiplier * (actual_home - expected_home)
        elo[home] += elo_change
        elo[away] -= elo_change
    df["home_elo"] = home_elos
    df["away_elo"] = away_elos
    df["elo_diff"] = df["home_elo"] - df["away_elo"]
    return df

# Rolling scoring form over last 5 games, shift(1) to ensure only past games are used
def add_rolling_team_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add rolling and differential features for teams."""
    df = df.copy()
    df["home_pts_last5"] = df.groupby("TEAM_ID_home")["PTS_home"].transform(
        lambda x: x.shift(1).rolling(5).mean()
    )

    df["away_pts_last5"] = df.groupby("TEAM_ID_away")["PTS_away"].transform(
        lambda x: x.shift(1).rolling(5).mean()
    )

    df["home_reb_last5"] = df.groupby("TEAM_ID_home")["REB_home"].transform(
        lambda x: x.shift(1).rolling(5).mean()
    )

    df["away_reb_last5"] = df.groupby("TEAM_ID_away")["REB_away"].transform(
        lambda x: x.shift(1).rolling(5).mean()
    )

    df["home_ast_last5"] = df.groupby("TEAM_ID_home")["AST_home"].transform(
        lambda x: x.shift(1).rolling(5).mean()
    )

    df["away_ast_last5"] = df.groupby("TEAM_ID_away")["AST_away"].transform(
        lambda x: x.shift(1).rolling(5).mean()
    )

    df["home_off_last10"] = df.groupby("TEAM_ID_home")["off_rating_home_game"].transform(
        lambda x: x.shift(1).rolling(10).mean()
    )

    df["away_off_last10"] = df.groupby("TEAM_ID_away")["off_rating_away_game"].transform(
        lambda x: x.shift(1).rolling(10).mean()
    )

    df["home_def_last10"] = df.groupby("TEAM_ID_home")["def_rating_home_game"].transform(
        lambda x: x.shift(1).rolling(10).mean()
    )

    df["away_def_last10"] = df.groupby("TEAM_ID_away")["def_rating_away_game"].transform(
        lambda x: x.shift(1).rolling(10).mean()
    )

    df["expected_pace"] = (df["poss_home"] + df["poss_away"]) / 2

    df["points_form_diff"] = df["home_pts_last5"] - df["away_pts_last5"]
    df["reb_form_diff"] = df["home_reb_last5"] - df["away_reb_last5"]
    df["ast_form_diff"] = df["home_ast_last5"] - df["away_ast_last5"]

    df["off_form_diff"] = df["home_off_last10"] - df["away_off_last10"]
    df["def_form_diff"] = df["home_def_last10"] - df["away_def_last10"]

    df["home_points_home_games"] = df.groupby("TEAM_ID_home")["PTS_home"].transform(
        lambda x: x.shift(1).rolling(10).mean()
    )

    df["away_points_away_games"] = df.groupby("TEAM_ID_away")["PTS_away"].transform(
        lambda x: x.shift(1).rolling(10).mean()
    )

    df["home_advantage_diff"] = (
        df["home_points_home_games"] - df["away_points_away_games"]
    )

    df["home_win_prev"] = df.groupby("TEAM_ID_home")["home_win"].shift(1)
    df["away_win_prev"] = df.groupby("TEAM_ID_away")["home_win"].shift(1)

    df["momentum_diff"] = df["home_win_prev"] - df["away_win_prev"]

    if {"NET_RATING_home", "NET_RATING_away"}.issubset(df.columns):
        df["official_net_rating_diff"] = (
            df["NET_RATING_home"] - df["NET_RATING_away"]
        )
    else:
        df["official_net_rating_diff"] = 0

    return df

# =========================================================
# BUILD FINAL FEATURE TABLE
# =========================================================
def build_feature_table(path: str) -> pd.DataFrame:
    """Build the final feature table for modeling."""
    df = load_data(path)
    df = simple_features(df)
    df = add_rest_schedule_features(df)
    df = add_elo_features(df)
    df = add_rolling_team_features(df)
    df = df.replace([np.inf, -np.inf], np.nan)
    
    df["rest_advantage"] = df.get("rest_diff", 0).fillna(0)
    df["elo_gap"] = (df.get("home_elo", 0) - df.get("away_elo", 0)).fillna(0)

    # ** Drop rows with missing values after feature engineering
    df = df.dropna().reset_index(drop=True)
    return df


# =========================================================
# EVALUATE TOTALS MODEL
# =========================================================
def train_totals_model(df: pd.DataFrame) -> None:
# Regression model to predict total points scored in a game (Future) 
    pass



# =========================================================
# EVALUATE WINNER MODELS
# =========================================================

def train_winner_model(df: pd.DataFrame) -> None:

# Trains model to predict game winners
# Features used:
    features_win = [
        "elo_diff",
        "points_form_diff",
        "reb_form_diff",
        "ast_form_diff",
        "rest_diff",
        "schedule_diff",
        "expected_pace",
        "momentum_diff",
        "home_advantage_diff",
        "off_form_diff",
        "def_form_diff",
        "official_net_rating_diff",
    ]

    X = df[features_win]

    y = df["home_win"]

# model is only trained on past data when predicting future games.
    split_idx = int(len(df) * 0.8)

    X_train = X.iloc[:split_idx]
    X_test = X.iloc[split_idx:]

    y_train = y.iloc[:split_idx]
    y_test = y.iloc[split_idx:]

# Random Forest Classifier notes:
# - Used because it handles nonlinear relationships well
# - Good for noisy data such like sports outcomes

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=6,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)

    print(f"Winner model accuracy: {acc:.3f}")

    # Save the trained model for script to reload and use without retraining.
    joblib.dump(model, MODEL_PATH)

    importances = model.feature_importances_

    print("\nFeature Importance:")
    for f, imp in sorted(zip(features_win, importances), key=lambda x: x[1], reverse=True):
        print(f"{f}: {imp:.4f}")
    print("Winner model saved to:", MODEL_PATH)

# =========================================================
# MAIN SCRIPT ENTRY POINT
# =========================================================
def main() -> None:
    """Main function to train and evaluate models, and save them."""
    df = build_feature_table(DATA_PATH)
    print("Final rows after feature engineering:", len(df))
    train_totals_model(df)
    train_winner_model(df)

if __name__ == "__main__":
    main()