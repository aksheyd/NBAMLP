import torch
import torch.optim as optim
from pathlib import Path

from data import build_dataset
from model import (
    NBAModel,
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    HIDDEN_SIZE_1,
    HIDDEN_SIZE_2,
)

MODEL_SAVE_PATH = Path("models")
MODEL_SAVE_PATH.mkdir(exist_ok=True)


class NBADataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def train_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    epochs=EPOCHS,
    save_path=None,
):
    best_val_acc = 0.0
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    print("\n" + "=" * 60)
    print("STARTING TRAINING")
    print("=" * 60)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()

            outputs = model(batch_X)

            one_hot = torch.nn.functional.one_hot(batch_y, num_classes=2).float()
            loss = torch.mean((outputs - one_hot) ** 2)

            model.backwards(outputs, batch_y)

            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()

        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                outputs = model(batch_X)

                one_hot = torch.nn.functional.one_hot(batch_y, num_classes=2).float()
                loss = torch.mean((outputs - one_hot) ** 2)

                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_total += batch_y.size(0)
                val_correct += (predicted == batch_y).sum().item()

        val_loss = val_loss / len(val_loader)
        val_acc = 100 * val_correct / val_total

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(
                f"Epoch [{epoch + 1:3d}/{epochs}] | "
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:5.2f}% | "
                f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:5.2f}%"
            )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            if save_path:
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_acc": val_acc,
                        "input_size": model.input_size,
                        "hidden_size_1": model.hidden_size_1,
                        "hidden_size_2": model.hidden_size_2,
                    },
                    save_path,
                )

    print("\n" + "=" * 60)
    print(f"TRAINING COMPLETE | Best Val Acc: {best_val_acc:.2f}%")
    print("=" * 60)

    return history


def evaluate_model(model, test_loader, meta_test):
    model.eval()
    correct = 0
    total = 0
    all_predictions = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            outputs = model(batch_X)

            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)

            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()

            all_predictions.extend(predicted.numpy())
            all_labels.extend(batch_y.numpy())
            all_probs.extend(probs[:, 1].numpy())

    accuracy = 100 * correct / total

    true_positives = sum(
        (pred == 1 and label == 1) for pred, label in zip(all_predictions, all_labels)
    )
    false_positives = sum(
        (pred == 1 and label == 0) for pred, label in zip(all_predictions, all_labels)
    )
    false_negatives = sum(
        (pred == 0 and label == 1) for pred, label in zip(all_predictions, all_labels)
    )
    true_negatives = sum(
        (pred == 0 and label == 0) for pred, label in zip(all_predictions, all_labels)
    )

    precision = (
        true_positives / (true_positives + false_positives)
        if (true_positives + false_positives) > 0
        else 0
    )
    recall = (
        true_positives / (true_positives + false_negatives)
        if (true_positives + false_negatives) > 0
        else 0
    )
    f1_score = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0
    )

    print("\n" + "=" * 60)
    print("TEST SET EVALUATION")
    print("=" * 60)
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1_score:.2f}")
    print("\nConfusion Matrix:")
    print(f"  True Positives:  {true_positives}")
    print(f"  False Positives: {false_positives}")
    print(f"  True Negatives:  {true_negatives}")
    print(f"  False Negatives: {false_negatives}")

    meta_test_reset = meta_test.reset_index(drop=True)
    print("\nChampion Predictions:")
    for i, label in enumerate(all_labels):
        if label == 1:
            team = meta_test_reset.iloc[i]["TEAM_NAME"]
            season = meta_test_reset.iloc[i]["SEASON"]
            prob = all_probs[i]
            pred = "[CORRECT]" if all_predictions[i] == 1 else "[MISSED]"
            print(f"  {pred} {season}: {team} (confidence: {prob:.2%})")

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "confusion_matrix": {
            "tp": true_positives,
            "fp": false_positives,
            "tn": true_negatives,
            "fn": false_negatives,
        },
    }


def predict_champion(model, season_data, scaler, feature_names, top_k=5):
    from data import engineer_features, select_features

    season_data = engineer_features(season_data)
    season_features = select_features(season_data)

    X = season_features[feature_names].values
    X_scaled = scaler.transform(X)

    model.eval()
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X_scaled)
        outputs = model(X_tensor)
        probs = torch.softmax(outputs, dim=1)
        champion_probs = probs[:, 1].numpy()

    results = season_features[["TEAM_NAME", "SEASON"]].copy()
    results["CHAMPIONSHIP_PROB"] = champion_probs
    results = results.sort_values("CHAMPIONSHIP_PROB", ascending=False)

    print("\n" + "=" * 60)
    print(f"TOP {top_k} CHAMPIONSHIP CONTENDERS")
    print("=" * 60)
    for i, row in results.head(top_k).iterrows():
        print(f"{row['TEAM_NAME']:25s} {row['CHAMPIONSHIP_PROB']:.2%}")

    return results


def main():
    print("\n" + "=" * 70)
    print("NBA CHAMPIONSHIP PREDICTION - TRAINING PIPELINE")
    print("=" * 70)

    print("\n[1/5] Loading dataset...")
    try:
        from data import load_processed_data, prepare_training_data

        df = load_processed_data()
        print("Loaded processed data from cache")
        data_splits = prepare_training_data(df)
    except FileNotFoundError:
        print("Building dataset from NBA API...")
        data_splits = build_dataset()

    X_train = data_splits["X_train"]
    y_train = data_splits["y_train"]
    X_val = data_splits["X_val"]
    y_val = data_splits["y_val"]
    X_test = data_splits["X_test"]
    y_test = data_splits["y_test"]
    meta_test = data_splits["meta_test"]

    input_size = X_train.shape[1]
    print(f"Input features: {input_size}")
    print(f"Training samples: {len(y_train)}")
    print(f"Validation samples: {len(y_val)}")
    print(f"Test samples: {len(y_test)}")

    print("\n[2/5] Creating data loaders...")
    train_dataset = NBADataset(X_train, y_train)
    val_dataset = NBADataset(X_val, y_val)
    test_dataset = NBADataset(X_test, y_test)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False
    )

    print("\n[3/5] Creating model...")
    model = NBAModel(
        input_size=input_size,
        hidden_size_1=HIDDEN_SIZE_1,
        hidden_size_2=HIDDEN_SIZE_2,
        output_size=2,
    )
    print(
        f"Model architecture: {input_size} -> {HIDDEN_SIZE_1} -> {HIDDEN_SIZE_2} -> 2"
    )

    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)

    print(f"\n[4/5] Training model for {EPOCHS} epochs...")
    model_path = MODEL_SAVE_PATH / "nba_champion_model.pt"
    history = train_model(
        model,
        train_loader,
        val_loader,
        optimizer,
        epochs=EPOCHS,
        save_path=model_path,
    )

    print("\n[5/5] Evaluating on test set...")
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint["model_state_dict"])

    metrics = evaluate_model(model, test_loader, meta_test)

    print("\n" + "=" * 70)
    print("TRAINING PIPELINE COMPLETE!")
    print("=" * 70)
    print(f"Model saved to: {model_path}")
    print(
        f"Final Test Accuracy: {metrics['accuracy']:.2f}% | F1 Score: {metrics['f1_score']:.2f}"
    )

    return model, history, metrics


if __name__ == "__main__":
    model, history, metrics = main()
