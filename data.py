import time
import pandas as pd
from pathlib import Path
from nba_api.stats.endpoints import leaguedashteamstats
from nba_api.stats.static import teams as nba_teams
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
START_YEAR = 1996
END_YEAR = 2024

CHAMPIONS = {
    "1995-96": "Chicago Bulls",
    "1996-97": "Chicago Bulls",
    "1997-98": "Chicago Bulls",
    "1998-99": "San Antonio Spurs",
    "1999-00": "Los Angeles Lakers",
    "2000-01": "Los Angeles Lakers",
    "2001-02": "Los Angeles Lakers",
    "2002-03": "San Antonio Spurs",
    "2003-04": "Detroit Pistons",
    "2004-05": "San Antonio Spurs",
    "2005-06": "Miami Heat",
    "2006-07": "San Antonio Spurs",
    "2007-08": "Boston Celtics",
    "2008-09": "Los Angeles Lakers",
    "2009-10": "Los Angeles Lakers",
    "2010-11": "Dallas Mavericks",
    "2011-12": "Miami Heat",
    "2012-13": "Miami Heat",
    "2013-14": "San Antonio Spurs",
    "2014-15": "Golden State Warriors",
    "2015-16": "Cleveland Cavaliers",
    "2016-17": "Golden State Warriors",
    "2017-18": "Golden State Warriors",
    "2018-19": "Toronto Raptors",
    "2019-20": "Los Angeles Lakers",
    "2020-21": "Milwaukee Bucks",
    "2021-22": "Golden State Warriors",
    "2022-23": "Denver Nuggets",
    "2023-24": "Boston Celtics",
}


def fetch_season_stats(season, season_type="Regular Season", use_cache=True):
    cache_file = RAW_DIR / f"{season}_{season_type.replace(' ', '_')}.csv"

    if use_cache and cache_file.exists():
        print(f"Loading {season} {season_type} from cache...")
        return pd.read_csv(cache_file)

    print(f"Fetching {season} {season_type} from NBA API...")

    try:
        stats = leaguedashteamstats.LeagueDashTeamStats(season=season)
        df = stats.get_data_frames()[0]

        df["SEASON"] = season
        df["SEASON_TYPE"] = season_type

        nba_team_names = [team["full_name"] for team in nba_teams.get_teams()]
        df = df[df["TEAM_NAME"].isin(nba_team_names)]

        df.to_csv(cache_file, index=False)
        time.sleep(1)

        return df

    except Exception as e:
        print(f"Error fetching {season}: {e}")
        return None


def collect_all_historical_data(start_year=START_YEAR, end_year=END_YEAR):
    all_data = []

    for year in range(start_year, end_year + 1):
        season = f"{year}-{str(year + 1)[-2:]}"

        if season not in CHAMPIONS and year < 2024:
            continue

        df = fetch_season_stats(season, "Regular Season")
        if df is not None:
            all_data.append(df)

    if not all_data:
        raise ValueError("No data collected!")

    combined_df = pd.concat(all_data, ignore_index=True)
    return combined_df


def fetch_current_season_stats(season="2024-25"):
    return fetch_season_stats(season, "Regular Season", use_cache=False)


def engineer_features(df):
    df = df.copy()

    if "WIN_PCT" not in df.columns:
        df["WIN_PCT"] = df["W"] / (df["W"] + df["L"])

    df["EFG_PCT"] = (df["FGM"] + 0.5 * df["FG3M"]) / df["FGA"]
    df["TS_PCT"] = df["PTS"] / (2 * (df["FGA"] + 0.44 * df["FTA"]))
    df["PPG"] = df["PTS"] / df["GP"]
    df["APG"] = df["AST"] / df["GP"]
    df["RPG"] = df["REB"] / df["GP"]
    df["TPG"] = df["TOV"] / df["GP"]
    df["AST_TO_RATIO"] = df["AST"] / (df["TOV"] + 1)
    df["FG3A_RATE"] = df["FG3A"] / df["FGA"]

    df = df.fillna(df.mean(numeric_only=True))

    return df


def select_features(df):
    feature_columns = [
        "WIN_PCT", "W", "L",
        "FG_PCT", "FG3_PCT", "FT_PCT", "EFG_PCT", "TS_PCT",
        "PPG", "PTS",
        "APG", "AST", "AST_TO_RATIO",
        "RPG", "OREB", "DREB", "REB",
        "STL", "BLK",
        "TPG", "TOV",
        "FG3M", "FG3A", "FG3A_RATE",
        "PLUS_MINUS",
    ]

    available_features = [col for col in feature_columns if col in df.columns]

    print(f"Selected {len(available_features)} features for training")

    return df[available_features + ["TEAM_NAME", "SEASON"]]


def add_champion_labels(df):
    df = df.copy()

    df["CHAMPION"] = df.apply(
        lambda row: 1 if CHAMPIONS.get(row["SEASON"]) == row["TEAM_NAME"] else 0,
        axis=1,
    )

    return df


def prepare_training_data(df, test_size=0.2, val_size=0.1, random_state=42):
    feature_cols = [
        col for col in df.columns if col not in ["CHAMPION", "TEAM_NAME", "SEASON"]
    ]
    X = df[feature_cols].values
    y = df["CHAMPION"].values
    metadata = df[["TEAM_NAME", "SEASON"]]

    X_temp, X_test, y_temp, y_test, meta_temp, meta_test = train_test_split(
        X, y, metadata, test_size=test_size, random_state=random_state, stratify=y
    )

    X_train, X_val, y_train, y_val, meta_train, meta_val = train_test_split(
        X_temp,
        y_temp,
        meta_temp,
        test_size=val_size / (1 - test_size),
        random_state=random_state,
        stratify=y_temp,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    import pickle

    scaler_path = PROCESSED_DIR / "scaler.pkl"
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    print(f"Scaler saved to {scaler_path}")

    return {
        "X_train": X_train_scaled,
        "y_train": y_train,
        "X_val": X_val_scaled,
        "y_val": y_val,
        "X_test": X_test_scaled,
        "y_test": y_test,
        "meta_train": meta_train,
        "meta_val": meta_val,
        "meta_test": meta_test,
        "scaler": scaler,
        "feature_names": feature_cols,
    }


def build_dataset(force_refresh=False):
    print("=" * 60)
    print("NBA CHAMPIONSHIP PREDICTION - DATA PIPELINE")
    print("=" * 60)

    print("\n[1/5] Collecting historical data...")
    if force_refresh:
        for file in RAW_DIR.glob("*.csv"):
            file.unlink()

    df_raw = collect_all_historical_data()
    print(f"Collected {len(df_raw)} team-seasons from {START_YEAR} to {END_YEAR}")

    print("\n[2/5] Engineering features...")
    df_engineered = engineer_features(df_raw)

    print("\n[3/5] Adding champion labels...")
    df_labeled = add_champion_labels(df_engineered)
    print(f"Champions: {df_labeled['CHAMPION'].sum()}")
    print(f"Non-champions: {(df_labeled['CHAMPION'] == 0).sum()}")

    print("\n[4/5] Selecting features...")
    df_features = select_features(df_labeled)
    df_features["CHAMPION"] = df_labeled["CHAMPION"]
    feature_cols = [
        col
        for col in df_features.columns
        if col not in ["CHAMPION", "TEAM_NAME", "SEASON"]
    ]
    print(f"Selected {len(feature_cols)} features")

    processed_file = PROCESSED_DIR / "nba_training_data.csv"
    df_features.to_csv(processed_file, index=False)
    print(f"Saved processed data to {processed_file}")

    print("\n[5/5] Preparing train/val/test splits...")
    data_splits = prepare_training_data(df_features)

    print("\nDataset shapes:")
    print(
        f"  Train: X={data_splits['X_train'].shape}, y={data_splits['y_train'].shape}"
    )
    print(f"  Val:   X={data_splits['X_val'].shape}, y={data_splits['y_val'].shape}")
    print(f"  Test:  X={data_splits['X_test'].shape}, y={data_splits['y_test'].shape}")

    print("\n" + "=" * 60)
    print("DATA PIPELINE COMPLETE!")
    print("=" * 60)

    return data_splits


def load_processed_data():
    processed_file = PROCESSED_DIR / "nba_training_data.csv"
    if not processed_file.exists():
        raise FileNotFoundError(
            f"Processed data not found at {processed_file}. Run build_dataset() first."
        )
    return pd.read_csv(processed_file)


def load_scaler():
    import pickle

    scaler_path = PROCESSED_DIR / "scaler.pkl"
    if not scaler_path.exists():
        raise FileNotFoundError(
            f"Scaler not found at {scaler_path}. Run build_dataset() first."
        )

    with open(scaler_path, "rb") as f:
        return pickle.load(f)


if __name__ == "__main__":
    data = build_dataset(force_refresh=False)

    print("\n\nSample training data:")
    print(data["meta_train"].head())
    print("\nFeature names:")
    print(data["feature_names"])
