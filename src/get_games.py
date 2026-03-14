from nba_api.stats.endpoints import leaguegamelog, leaguedashteamstats
import pandas as pd
import time

# SEASONS TO DOWNLOAD (2019-2024)
seasons = ["2023-24", "2022-23", "2021-22", "2020-21", "2019-20"]

all_games = []
all_ratings = []

# get game logs and team ratings for each season, with error handling and API rate limit protection
for season in seasons:

    print(f"Downloading season: {season}")

    try:
        # -------------------------
        # GAME LOGS
        # -------------------------

        games = leaguegamelog.LeagueGameLog(
            season=season, season_type_all_star="Regular Season"
        )

        games_df = games.get_data_frames()[0]
        games_df["SEASON"] = season

        all_games.append(games_df)

        # -------------------------
        # TEAM RATINGS
        # -------------------------

        ratings = leaguedashteamstats.LeagueDashTeamStats(
            season=season,
            season_type_all_star="Regular Season",
            measure_type_detailed_defense="Advanced",
            per_mode_detailed="Per100Possessions",
        )

        ratings_df = ratings.get_data_frames()[0]

        ratings_df = ratings_df[
            ["TEAM_ID", "TEAM_NAME", "OFF_RATING", "DEF_RATING", "NET_RATING", "PACE"]
        ]

        ratings_df["SEASON"] = season

        all_ratings.append(ratings_df)

        # API rate limit protection
        time.sleep(1.2)

    except Exception as e:

        print("Error downloading season:", season)
        print(e)

# =============================
# COMBINE DATA
# =============================

games = pd.concat(all_games, ignore_index=True)
ratings = pd.concat(all_ratings, ignore_index=True)

# =============================
# MERGE RATINGS INTO GAMES
# =============================

games = games.merge(
    ratings, left_on=["TEAM_ID", "SEASON"], right_on=["TEAM_ID", "SEASON"], how="left"
)

# =============================
# SAVE DATASET
# =============================

games.to_csv("data/nba_games_multi_season.csv", index=False)

print("Dataset saved!")
print("Rows:", len(games))
print(games.head())
