import torch
from data import (
    fetch_current_season_stats,
    engineer_features,
    select_features,
    load_scaler,
)
from model import NBAModel


def predict_current_season(
    model_path="models/nba_champion_model.pt", season="2024-25", top_k=10
):
    print("=" * 70)
    print(f"NBA CHAMPIONSHIP PREDICTION - {season} SEASON")
    print("=" * 70)

    print("\n[1/3] Loading trained model...")
    checkpoint = torch.load(model_path)
    model = NBAModel(
        input_size=checkpoint["input_size"],
        hidden_size_1=checkpoint["hidden_size_1"],
        hidden_size_2=checkpoint["hidden_size_2"],
        output_size=2,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"Model loaded with validation accuracy: {checkpoint['val_acc']:.2f}%")

    print("\n[2/3] Loading feature scaler...")
    scaler = load_scaler()

    print(f"\n[3/3] Fetching {season} statistics...")
    current_data = fetch_current_season_stats(season)

    if current_data is None or len(current_data) == 0:
        print(f"Error: Could not fetch data for {season} season")
        return

    print(f"Fetched data for {len(current_data)} teams")

    current_data = engineer_features(current_data)
    current_features = select_features(current_data)

    feature_cols = [
        col for col in current_features.columns if col not in ["TEAM_NAME", "SEASON"]
    ]

    X = current_features[feature_cols].values
    X_scaled = scaler.transform(X)

    with torch.no_grad():
        X_tensor = torch.FloatTensor(X_scaled)
        outputs = model(X_tensor)
        probs = torch.softmax(outputs, dim=1)
        champion_probs = probs[:, 1].numpy()

    results = current_features[["TEAM_NAME"]].copy()
    results["CHAMPIONSHIP_PROB"] = champion_probs
    results = results.sort_values("CHAMPIONSHIP_PROB", ascending=False)

    print("\n" + "=" * 70)
    print(f"TOP {top_k} CHAMPIONSHIP CONTENDERS FOR {season}")
    print("=" * 70)
    print(f"{'Rank':<6} {'Team':<30} {'Probability':>15}")
    print("-" * 70)

    for rank, (idx, row) in enumerate(results.head(top_k).iterrows(), 1):
        print(f"{rank:<6} {row['TEAM_NAME']:<30} {row['CHAMPIONSHIP_PROB']:>14.2%}")

    print("=" * 70)

    return results


if __name__ == "__main__":
    predictions = predict_current_season()
