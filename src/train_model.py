import numpy as np
import pandas as pd
import os
import joblib

from sklearn.ensemble import VotingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "nba_games_multi_season.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "winner_model.pkl")


RANDOM_STATE = 42
N_SPLITS = 5

TEAM_ID_TO_NAME = {
    1610612737: "atlanta hawks",
    1610612738: "boston celtics",
    1610612739: "cleveland cavaliers",
    1610612740: "new orleans pelicans",
    1610612741: "chicago bulls",
    1610612742: "dallas mavericks",
    1610612743: "denver nuggets",
    1610612744: "golden state warriors",
    1610612745: "houston rockets",
    1610612746: "los angeles clippers",
    1610612747: "los angeles lakers",
    1610612748: "miami heat",
    1610612749: "milwaukee bucks",
    1610612750: "minnesota timberwolves",
    1610612751: "brooklyn nets",
    1610612752: "new york knicks",
    1610612753: "orlando magic",
    1610612754: "indiana pacers",
    1610612755: "philadelphia 76ers",
    1610612756: "phoenix suns",
    1610612757: "portland trail blazers",
    1610612758: "sacramento kings",
    1610612759: "san antonio spurs",
    1610612760: "oklahoma city thunder",
    1610612761: "toronto raptors",
    1610612762: "utah jazz",
    1610612763: "memphis grizzlies",
    1610612764: "washington wizards",
    1610612765: "detroit pistons",
    1610612766: "charlotte hornets",
}

TEAM_ABBREV_MAP = {
    "atl": "atlanta hawks",
    "bos": "boston celtics",
    "cle": "cleveland cavaliers",
    "nop": "new orleans pelicans",
    "chi": "chicago bulls",
    "dal": "dallas mavericks",
    "den": "denver nuggets",
    "gs": "golden state warriors",
    "gsw": "golden state warriors",
    "hou": "houston rockets",
    "lac": "los angeles clippers",
    "lal": "los angeles lakers",
    "mia": "miami heat",
    "mil": "milwaukee bucks",
    "min": "minnesota timberwolves",
    "bkn": "brooklyn nets",
    "ny": "new york knicks",
    "nyk": "new york knicks",
    "orl": "orlando magic",
    "ind": "indiana pacers",
    "phi": "philadelphia 76ers",
    "phx": "phoenix suns",
    "por": "portland trail blazers",
    "sac": "sacramento kings",
    "sa": "san antonio spurs",
    "sas": "san antonio spurs",
    "okc": "oklahoma city thunder",
    "tor": "toronto raptors",
    "uta": "utah jazz",
    "mem": "memphis grizzlies",
    "wsh": "washington wizards",
    "det": "detroit pistons",
    "cha": "charlotte hornets",
}

# after engineernig  features and thier weights, I will trian the model to predict the winner and "probability".
# Then I'll covert vegas odds to probability and see if my model predicts more of a probability, or "edge"

# ML project / learning purposes **

#WIN PROB > MARGIN (BY " 4.5 points"), and > UPSET ALERT **

#ML model to predict NBA game outcomes

# Steps:
# 1. Load and clean NBA game data
# 2. Engineer predictive features
# 3. Train classification models to predict 
# 4. Evaluate accuracy on a time-based holdout set
# 5. Save the trained model for use in live predictions

# use information available *before* each game, making it suitable for predicting future games.

def load_data(path: str) -> pd.DataFrame:
# Load and clean NBA games data
    #Load
    df = pd.read_csv(path).copy()
    df.columns = df.columns.str.strip()

 # Dates + sorting
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"]).dt.normalize()
    df = df.sort_values(["GAME_DATE", "GAME_ID"]).reset_index(drop=True)

 # identify home and away teams in game
    home = df[df["MATCHUP"].str.contains("vs.", na=False)].copy()
    away = df[df["MATCHUP"].str.contains("@", na=False)].copy()

    games = home.merge(
        away,
        on="GAME_ID",
        suffixes=("_home", "_away"),
        how="inner",
    )

    games["GAME_DATE_home"] = pd.to_datetime(games["GAME_DATE_home"]).dt.normalize()

    #TARGETS AND LABELS -- 
    games["home_win"] = (games["PTS_home"] > games["PTS_away"]).astype(int) # Output: 1 for win , 0 for loss
    games["total_points"] = games["PTS_home"] + games["PTS_away"] # Output total score of games 
    games["margin_home"] = games["PTS_home"] - games["PTS_away"] #outoput by how much winner will win

    # Cleaning data, only games with realistic points
    games = games[(games["total_points"] > 150) & (games["total_points"] < 300)].reset_index(drop=True)

    #Final sort
    games = games.sort_values(["GAME_DATE_home", "GAME_ID"]).reset_index(drop=True)

    return games

def load_clean_odds():
    import kagglehub

    path = kagglehub.dataset_download(
        "cviaxmiwnptr/nba-betting-data-october-2007-to-june-2024"
    )

    print("\nDataset path:", path)
    print("Files:", os.listdir(path))


    csv_path = os.path.join(path, "nba_2008-2025.csv") #changed dataset 3/19/2026 to include begas odds
    df = pd.read_csv(csv_path)

    df.columns = df.columns.str.lower().str.strip()
    print("\nOdds columns:")
    print(df.columns.tolist())
    print(df.head())

    return df

 # ## ** base feature base: (simplified 3/17/2026)
# 1. Team Overall Rank / Rating - Who is the better team overall ?  :: check into OFF_RATING, DEF_RATING, AND NET_RATING** use net first
# 2. Team recent form (last 5 games) - Who is playing better recently ? ::  rr
# 3. Home/Away performance splits How or away ? 
#4.  Rest days since last game - Which team is more rested ?
# 5. Injuries - seperate API fetch injuries*** 
# 6. Pace of play (estimated possessions) - Which team plays faster :: PACE

def engineer_features(games):
    df = games.copy()
    # net rating
    df["power_diff"] = df["NET_RATING_home"] - df["NET_RATING_away"]

    # mathup diff
    df["matchup_diff"] = ((df["OFF_RATING_home"] - df["DEF_RATING_away"]) -(df["OFF_RATING_away"] - df["DEF_RATING_home"]))
    df["win_pct_last10_diff"] = df["home_win_last10"] - df["away_win_last10"]
    # Four factors of influence from Dean Oliver - ( biggest influences of basketball game )
    # 1. Shooting (~40%) → eFG%
    df["efg_home"] = ((df["FGM_home"] + 0.5 * df["FG3M_home"]) / df["FGA_home"].replace(0, np.nan))
    df["efg_away"] = ((df["FGM_away"] + 0.5 * df["FG3M_away"]) /df["FGA_away"].replace(0, np.nan))
    df["efg_diff"] = df["efg_home"] - df["efg_away"]
    # 2.Turnovers (~25%)
    df["tov_diff"] = df["TOV_away"] - df["TOV_home"]
    # (positive = home turns it over less → good)
    # 3. Rebounding  (~20%)
    df["reb_diff"] = df["REB_home"] - df["REB_away"]
    # 4. Free throws influence (~15%)
    df["ft_rate_home"] = df["FTA_home"] / df["FGA_home"].replace(0, np.nan)
    df["ft_rate_away"] = df["FTA_away"] / df["FGA_away"].replace(0, np.nan)
    df["ft_rate_diff"] = df["ft_rate_home"] - df["ft_rate_away"]

    # defense (positive = home better defense)
    df["def_rating_diff"] = df["DEF_RATING_away"] - df["DEF_RATING_home"]
    # posessions ? which team had more  
    df["poss_diff"] = df["reb_diff"] + df["tov_diff"]
    #expected pace of game
    df["pace_diff"] = df["PACE_home"] - df["PACE_away"]
    # home advantage ?
    df["home_team"] = 1

    return df

# Rest and Schedule Features
def add_restdays(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Sort properly
    df = df.sort_values(["TEAM_ID_home", "GAME_DATE_home"])

    # -------------------------
    # REST DAYS
    # -------------------------
    home_last = df.groupby("TEAM_ID_home")["GAME_DATE_home"].shift(1)
    away_last = df.groupby("TEAM_ID_away")["GAME_DATE_home"].shift(1)

    df["home_rest_days"] = (df["GAME_DATE_home"] - home_last).dt.days
    df["away_rest_days"] = (df["GAME_DATE_home"] - away_last).dt.days

    df["rest_diff"] = df["home_rest_days"] - df["away_rest_days"]

    df = df.sort_values("GAME_DATE_home").reset_index(drop=True)
    # HOME
    df["home_games_last7"] = 0.0
    for team, idx in df.groupby("TEAM_ID_home").groups.items():
        team_df = df.loc[idx].sort_values("GAME_DATE_home")

        counts = (
            pd.Series(1, index=team_df["GAME_DATE_home"])
            .shift(1)
            .rolling("7D")
            .sum()
        )
        df.loc[team_df.index, "home_games_last7"] = counts.values

# AWAY
    df["away_games_last7"] = 0.0
    for team, idx in df.groupby("TEAM_ID_away").groups.items():
        team_df = df.loc[idx].sort_values("GAME_DATE_home")

        counts = (
            pd.Series(1, index=team_df["GAME_DATE_home"])
            .shift(1)
            .rolling("7D")
            .sum()
        )
        df.loc[team_df.index, "away_games_last7"] = counts.values

    df["home_games_last7"] = df["home_games_last7"].fillna(0)
    df["away_games_last7"] = df["away_games_last7"].fillna(0)
    df["schedule_diff"] = df["home_games_last7"] - df["away_games_last7"]

    return df

def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Ensure proper order
    df = df.sort_values(["GAME_DATE_home", "GAME_ID"])
    # Last 5 Form
    df["home_pts_last5"] = df.groupby("TEAM_ID_home")["PTS_home"].transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
    df["away_pts_last5"] = df.groupby("TEAM_ID_away")["PTS_away"].transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
    df["home_reb_last5"] = df.groupby("TEAM_ID_home")["REB_home"].transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
    df["away_reb_last5"] = df.groupby("TEAM_ID_away")["REB_away"].transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
    # Last10
    df["home_off_last10"] = df.groupby("TEAM_ID_home")["OFF_RATING_home"].transform(lambda x: x.shift(1).rolling(10, min_periods=1).mean())
    df["away_off_last10"] = df.groupby("TEAM_ID_away")["OFF_RATING_away"].transform(lambda x: x.shift(1).rolling(10, min_periods=1).mean())
    df["home_def_last10"] = df.groupby("TEAM_ID_home")["DEF_RATING_home"].transform(lambda x: x.shift(1).rolling(10, min_periods=1).mean())
    df["away_def_last10"] = df.groupby("TEAM_ID_away")["DEF_RATING_away"].transform(lambda x: x.shift(1).rolling(10, min_periods=1).mean())
    df["points_form_diff"] = df["home_pts_last5"] - df["away_pts_last5"]
    df["reb_form_diff"] = df["home_reb_last5"] - df["away_reb_last5"]
    df["off_form_diff"] = df["home_off_last10"] - df["away_off_last10"]
    df["def_form_diff"] = df["home_def_last10"] - df["away_def_last10"]
    #momentum
    df["home_win_last5"] = df.groupby("TEAM_ID_home")["home_win"].transform(
        lambda x: x.shift(1).rolling(5, min_periods=1).mean()
    )
    df["away_win_last5"] = df.groupby("TEAM_ID_away")["home_win"].transform(
        lambda x: x.shift(1).rolling(5, min_periods=1).mean()
    )
    df["momentum_diff"] = df["home_win_last5"] - df["away_win_last5"]
    # Shooting
    df["home_efg_last10"] = df.groupby("TEAM_ID_home")["efg_home"].transform(lambda x: x.shift(1).rolling(10).mean())
    df["away_efg_last10"] = df.groupby("TEAM_ID_away")["efg_away"].transform(lambda x: x.shift(1).rolling(10).mean())
    df["efg_diff_roll"] = df["home_efg_last10"] - df["away_efg_last10"]
    # Turnovers
    df["home_tov_last10"] = df.groupby("TEAM_ID_home")["TOV_home"].transform(lambda x: x.shift(1).rolling(10).mean())
    df["away_tov_last10"] = df.groupby("TEAM_ID_away")["TOV_away"].transform(lambda x: x.shift(1).rolling(10).mean())
    df["tov_diff_roll"] = df["away_tov_last10"] - df["home_tov_last10"]
    # Rebounding
    df["home_reb_last10"] = df.groupby("TEAM_ID_home")["REB_home"].transform(lambda x: x.shift(1).rolling(10).mean())
    df["away_reb_last10"] = df.groupby("TEAM_ID_away")["REB_away"].transform(lambda x: x.shift(1).rolling(10).mean())
    df["reb_diff_roll"] = df["home_reb_last10"] - df["away_reb_last10"]
    return df

def build_feature_table(path: str) -> pd.DataFrame:
    df = load_data(path)
    df = engineer_features(df)
    df = add_restdays(df)
    df = add_rolling_features(df)

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(0)

    return df

def get_feature_list() -> list[str]:
    return [
        "power_diff",
        "matchup_diff",
        "efg_diff_roll",
        "tov_diff_roll",
        "reb_diff_roll",
        "pace_diff",
        "rest_diff",
        "schedule_diff",
        "points_form_diff",
        "reb_form_diff",
        "off_form_diff",
        "def_form_diff",
        "momentum_diff",
        "vegas_home_prob_no_vig"
    ]


def time_split(df: pd.DataFrame, split_date: str):
    train = df[df["GAME_DATE_home"] < split_date].copy()
    test = df[df["GAME_DATE_home"] >= split_date].copy()
    return train, test

def build_ensemble():

    lr = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000)
    )

    xgb = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
        eval_metric="logloss"
    )

    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=6,
        random_state=RANDOM_STATE
    )

    model = VotingClassifier(
        estimators=[
            ("lr", lr),
            ("xgb", xgb),
            ("rf", rf)
        ],
        voting="soft",
        weights=[1, 3, 2]
    )

    return model

def evaluate_classifier(model, X_train, y_train, X_test, y_test, label: str):
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    metrics = {
        "model": label,
        "accuracy": accuracy_score(y_test, preds),
        "log_loss": log_loss(y_test, probs),
        "roc_auc": roc_auc_score(y_test, probs),
    }

    print(f"\n{label}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Log Loss: {metrics['log_loss']:.4f}")
    print(f"ROC AUC : {metrics['roc_auc']:.4f}")

    return model, preds, probs, metrics


def walk_forward_cv(df: pd.DataFrame, features: list[str], target: str = "home_win"):

    df = df.sort_values("GAME_DATE_home").reset_index(drop=True)
    X = df[features].copy()
    y = df[target].copy()

    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    rows = []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X), start=1):

        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]


        model = build_ensemble()
        model.fit(X_train, y_train)

        probs = model.predict_proba(X_test)[:, 1]
        preds = model.predict(X_test)

        rows.append({
            "fold": fold,
            "accuracy": accuracy_score(y_test, preds),
            "log_loss": log_loss(y_test, probs),
            "roc_auc": roc_auc_score(y_test, probs),

            "avg_prob": probs.mean(),
            "std_prob": probs.std(),
        })

    results = pd.DataFrame(rows)

    print("\nWalk-forward CV results")
    print(results)
    print("\nAverages")
    print(results.mean(numeric_only=True))
    print("\nStability Check: low == stable model")
    print("Accuracy std:", results["accuracy"].std())
    print("Log loss std:", results["log_loss"].std())

    return results


def backtest_strategy(df, probs, original_games, threshold=0.05):
    df = df.copy().reset_index(drop=True)

    
    df["model_prob"] = pd.Series(probs)
    df["edge"] = df["model_prob"] - df["vegas_home_prob_no_vig"]
    bets = df[df["edge"] > threshold].copy()   # ONLY positive edge
    bets["prediction"] = (bets["model_prob"] > 0.5).astype(int)
    print("\nTop 10 bets:")
    print(bets[["model_prob", "vegas_home_prob_no_vig", "edge"]]
      .sort_values("edge", ascending=False)
      .head(10))
    wins = (bets["prediction"] == bets["home_win"]).sum()
    total = len(bets)

    print("\nBACKTEST RESULTS (REAL EDGE)")
    print("Total Bets:", total)
    print("Win Rate :", wins / total if total > 0 else 0)
    print("Avg Edge :", bets["edge"].mean() if total > 0 else 0)
    print("Bet %    :", len(bets) / original_games)

    return bets

def american_to_prob(odds):
    if odds < 0:
        return -odds / (-odds + 100)
    else:
        return 100 / (odds + 100)

def merge_odds(games: pd.DataFrame, odds: pd.DataFrame) -> pd.DataFrame:
    
    df = games.copy()
    odds = odds.copy()
    odds.columns = odds.columns.str.lower().str.strip()
    odds_date_col = "date"
    odds_home_team_col = "home"
    odds_away_team_col = "away"
    odds_home_ml_col = "moneyline_home"

    
    df["GAME_DATE_home"] = pd.to_datetime(df["GAME_DATE_home"]).dt.normalize()
    odds[odds_date_col] = pd.to_datetime(odds[odds_date_col]).dt.normalize()

    df["home_team_name"] = df["TEAM_ID_home"].map(TEAM_ID_TO_NAME).str.lower()
    df["away_team_name"] = df["TEAM_ID_away"].map(TEAM_ID_TO_NAME).str.lower()

    odds[odds_home_team_col] = odds[odds_home_team_col].str.lower().map(TEAM_ABBREV_MAP).fillna(odds[odds_home_team_col])
    odds[odds_away_team_col] = odds[odds_away_team_col].str.lower().map(TEAM_ABBREV_MAP).fillna(odds[odds_away_team_col])

    # merged
    merged = df.merge(
        odds,
        left_on=["home_team_name", "away_team_name", "GAME_DATE_home"],
        right_on=[odds_home_team_col, odds_away_team_col, odds_date_col],
        how="left"
    )
    print("\nMerge success rate:", merged[odds_home_ml_col].notna().mean())
    merged["vegas_home_prob_no_vig"] = merged[odds_home_ml_col].apply(
        lambda x: american_to_prob(x) if pd.notna(x) else np.nan
    )
    return merged

def run_pipeline():
    games = build_feature_table(DATA_PATH)
    games = games.sort_values("GAME_DATE_home")

    print("\nAfter dedup:", games.shape)
    odds = load_clean_odds()
    games = merge_odds(games, odds)
    print("\nAfter odds merge:", games.shape)
    games = games.dropna(subset=["vegas_home_prob_no_vig"])
    print("\nAfter dropping missing vegas:", games.shape)
    print("\nOdds column check:")
    print(games["vegas_home_prob_no_vig"].describe())

    print("\nSample merged odds rows:")
    print(
        games[["GAME_ID", "TEAM_ID_home", "TEAM_ID_away", "vegas_home_prob_no_vig"]]
        .head(10)
    )

    FEATURES = get_feature_list()
    TARGET = "home_win"

    print("\nDataset shape:", games.shape)
    print("Feature sample:")
    print(games[FEATURES + [TARGET]].head())

    train, test = time_split(games, split_date="2021-01-01")
    X_train = train[FEATURES]
    y_train = train[TARGET]
    X_test = test[FEATURES]
    y_test = test[TARGET]

    print("\nTrain shape:", X_train.shape)
    print("Test shape :", X_test.shape)

    baseline = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)
    )
    baseline_model, baseline_preds, baseline_probs, baseline_metrics = evaluate_classifier(
        baseline, X_train, y_train, X_test, y_test, "Baseline Logistic Regression"
    )

    xgb = XGBClassifier(
    n_estimators=500,
    max_depth=3,
    learning_rate=0.02,
    subsample=0.7,
    colsample_bytree=0.7,
    min_child_weight=3,
    gamma=0.1,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=RANDOM_STATE,
    eval_metric="logloss",
)

    xgb_model, xgb_preds, xgb_probs, xgb_metrics = evaluate_classifier(
        xgb, X_train, y_train, X_test, y_test, "XGBoost"
    )

    ensemble = build_ensemble()
    ensemble_model, ensemble_preds, ensemble_probs, ensemble_metrics = evaluate_classifier(
        ensemble, X_train, y_train, X_test, y_test, "Ensemble Model"
    )

    print("Baseline Accuracy:", baseline_metrics["accuracy"])
    print("XGB Accuracy     :", xgb_metrics["accuracy"])
    print("Ensemble Accuracy:", ensemble_metrics["accuracy"])
    print("\nPrediction sanity check:")
    print("Min prob:", ensemble_probs.min())
    print("Max prob:", ensemble_probs.max())
    print("Mean prob:", ensemble_probs.mean())
    print("\nCalibration Checks")
    print("Avg predicted prob:", ensemble_probs.mean())
    print("Actual win rate   :", y_test.mean())
    test = test.copy()
    test["model_prob"] = ensemble_probs
    games = games.copy()
    games["model_prob"] = np.nan
    games.loc[test.index, "model_prob"] = test["model_prob"]
    games["edge"] = games["model_prob"] - games["vegas_home_prob_no_vig"]
    original_games = len(games)
    modeling_games = games.copy()
    games = games[(games["vegas_home_prob_no_vig"] > 0.10) & (games["vegas_home_prob_no_vig"] < 0.90)]
    games = games.dropna(subset=["model_prob"])

    bets = games[
        (games["edge"] > 0.05) &
        (games["vegas_home_prob_no_vig"] > 0.20) &
        (games["vegas_home_prob_no_vig"] < 0.80)
    ]
    print("\nEDGE ANALYSIS (FILTERED)")
    print("Total Bets:", len(bets))
    print("Model avg prob:", bets["model_prob"].mean())
    print("Vegas avg prob:", bets["vegas_home_prob_no_vig"].mean())
    print("Avg edge:", bets["edge"].mean())

    cv_results = walk_forward_cv(
        modeling_games.dropna(subset=["model_prob"]),
        FEATURES,
        TARGET
    )
    importance = pd.Series(
        xgb_model.feature_importances_, index=FEATURES
    ).sort_values(ascending=False)

    print("\nFeature Importance")
    print(importance)

    best_model = ensemble_model
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(best_model, MODEL_PATH)
    print(f"\nSaved model to: {MODEL_PATH}")

    backtest_strategy(bets, bets["model_prob"].values, original_games)

    results = {
        "games": games,
        "train": train,
        "test": test,
        "baseline_metrics": baseline_metrics,
        "xgb_metrics": xgb_metrics,
        "ensemble_metrics": ensemble_metrics,
        "cv_results": cv_results,
        "feature_importance": importance,
        "probs": ensemble_probs,
    }
    return results


if __name__ == "__main__":
    results = run_pipeline()