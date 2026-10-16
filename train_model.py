"""
train_model.py
==============
Trains an environmental Room Occupancy Detection model using:
- Dataset: Occupancy.csv (UCI Room Occupancy dataset)
- Input Features: ['Temperature', 'Humidity', 'Light'] (strictly matching available ESP32 DHT11 & LDR sensors)
- Target: Occupancy (0 = Unoccupied, 1 = Occupied)
- Data Split: Chronological time-series split (80% train, 20% test) to prevent temporal data leakage.
- Models: Baseline (Dummy, Logistic Regression) and Random Forest Classifier.
- Artifacts Saved:
    - models/occupancy_rf_model.joblib (Model pipeline + ordered feature names + metadata)
    - models/metrics.json (Genuine test metrics for API/Dashboard)
    - static/img/confusion_matrix.png (Visualization of evaluation matrix)
"""

import json
import os
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FEATURE_COLUMNS = ['Temperature', 'Humidity', 'Light']
TARGET_COLUMN = 'Occupancy'
DATASET_PATH = 'Occupancy.csv'
MODEL_SAVE_PATH = os.path.join('models', 'occupancy_rf_model.joblib')
METRICS_SAVE_PATH = os.path.join('models', 'metrics.json')
PLOT_SAVE_PATH = os.path.join('static', 'img', 'confusion_matrix.png')


def load_and_preprocess_data(csv_path: str):
    """Load dataset, sort chronologically, and validate schema."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found at {csv_path}")

    print(f"Loading dataset from: {csv_path}")
    df = pd.read_csv(csv_path)

    # Convert date to datetime and sort to maintain temporal order
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)

    print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"Missing values across features: {df[FEATURE_COLUMNS].isnull().sum().to_dict()}")

    return df


def time_based_split(df: pd.DataFrame, train_ratio: float = 0.8):
    """
    Split time-series data chronologically to prevent temporal data leakage.
    Earlier 80% used for training, subsequent 20% used for testing.
    """
    split_index = int(len(df) * train_ratio)
    train_df = df.iloc[:split_index].copy()
    test_df = df.iloc[split_index:].copy()

    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df[TARGET_COLUMN]
    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df[TARGET_COLUMN]

    split_info = {
        'total_samples': len(df),
        'train_samples': len(train_df),
        'test_samples': len(test_df),
        'train_start_date': str(train_df['date'].min()),
        'train_end_date': str(train_df['date'].max()),
        'test_start_date': str(test_df['date'].min()),
        'test_end_date': str(test_df['date'].max()),
        'train_class_distribution': train_df[TARGET_COLUMN].value_counts().to_dict(),
        'test_class_distribution': test_df[TARGET_COLUMN].value_counts().to_dict(),
    }

    print("\n--- CHRONOLOGICAL SPLIT DETAILS ---")
    print(f"Train period: {split_info['train_start_date']} -> {split_info['train_end_date']} ({split_info['train_samples']} samples)")
    print(f"Test period:  {split_info['test_start_date']} -> {split_info['test_end_date']} ({split_info['test_samples']} samples)")
    print(f"Train label balance: {split_info['train_class_distribution']}")
    print(f"Test label balance:  {split_info['test_class_distribution']}")

    return X_train, y_train, X_test, y_test, split_info


def evaluate_model(name: str, y_true, y_pred, y_prob=None):
    """Compute and format standard classification metrics."""
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, y_prob) if y_prob is not None else None

    metrics = {
        'model_name': name,
        'accuracy': float(acc),
        'precision_class1': float(prec),
        'recall_class1': float(rec),
        'f1_score_class1': float(f1),
        'roc_auc': float(auc) if auc is not None else None,
        'confusion_matrix': confusion_matrix(y_true, y_pred).tolist(),
    }

    print(f"\n--- {name.upper()} RESULTS ---")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-Score : {f1:.4f}")
    if auc is not None:
        print(f"ROC-AUC  : {auc:.4f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

    return metrics


def plot_confusion_matrix(cm, save_path):
    """Generate and save an aesthetically styled confusion matrix plot."""
    fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
    fig.patch.set_facecolor('#0f172a')
    ax.set_facecolor('#0f172a')

    # Color map
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(colors='#94a3b8')

    classes = ['Unoccupied (0)', 'Occupied (1)']
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(classes, color='#cbd5e1', fontsize=11, fontweight='600')
    ax.set_yticklabels(classes, color='#cbd5e1', fontsize=11, fontweight='600')

    # Values in boxes
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            color = "#0f172a" if val > thresh else "#f8fafc"
            ax.text(j, i, f"{val:,}\n({val/cm.sum()*100:.1f}%)",
                    ha="center", va="center",
                    color=color, fontsize=12, fontweight='bold')

    ax.set_title("Test Confusion Matrix (Time-Split)\nRandom Forest [Temp, Humidity, Light]",
                 color='#f8fafc', fontsize=12, fontweight='bold', pad=15)
    ax.set_ylabel('Ground Truth Label', color='#94a3b8', fontsize=11, labelpad=10)
    ax.set_xlabel('Predicted Label', color='#94a3b8', fontsize=11, labelpad=10)

    # Spine styling
    for spine in ax.spines.values():
        spine.set_color('#334155')

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    print(f"Confusion matrix plot saved to: {save_path}")


def main():
    print("=" * 60)
    print("Smart Room Occupancy Detection - ML Training Pipeline")
    print("Environmental Features: ['Temperature', 'Humidity', 'Light']")
    print("=" * 60)

    # 1. Load data
    df = load_and_preprocess_data(DATASET_PATH)

    # 2. Time-based chronological split (prevents temporal data leakage)
    X_train, y_train, X_test, y_test, split_info = time_based_split(df, train_ratio=0.8)

    # 3. Baseline 1: Dummy Majority Class Classifier
    dummy = DummyClassifier(strategy='most_frequent')
    dummy.fit(X_train, y_train)
    dummy_preds = dummy.predict(X_test)
    dummy_metrics = evaluate_model("Dummy Majority Baseline", y_test, dummy_preds)

    # 4. Baseline 2: Logistic Regression (Standardized)
    log_reg = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(random_state=42))
    ])
    log_reg.fit(X_train, y_train)
    lr_preds = log_reg.predict(X_test)
    lr_probs = log_reg.predict_proba(X_test)[:, 1]
    lr_metrics = evaluate_model("Logistic Regression Baseline", y_test, lr_preds, lr_probs)

    # 5. Primary Model: Random Forest Classifier
    rf_clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=12,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    rf_clf.fit(X_train, y_train)
    rf_preds = rf_clf.predict(X_test)
    rf_probs = rf_clf.predict_proba(X_test)[:, 1]
    rf_metrics = evaluate_model("Random Forest Model", y_test, rf_preds, rf_probs)

    # Feature importances
    feature_importances = {
        col: float(imp) for col, imp in zip(FEATURE_COLUMNS, rf_clf.feature_importances_)
    }
    print("\n--- FEATURE IMPORTANCES ---")
    for feat, imp in feature_importances.items():
        print(f"  {feat:15s}: {imp * 100:6.2f}%")

    # 6. Generate confusion matrix plot
    cm = np.array(rf_metrics['confusion_matrix'])
    plot_confusion_matrix(cm, PLOT_SAVE_PATH)

    # 7. Package and save model bundle
    model_bundle = {
        'model': rf_clf,
        'feature_names': FEATURE_COLUMNS,
        'target_name': TARGET_COLUMN,
        'feature_importances': feature_importances,
        'metrics': rf_metrics,
        'split_info': split_info,
        'baseline_metrics': {
            'dummy': dummy_metrics,
            'logistic_regression': lr_metrics
        }
    }

    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    joblib.dump(model_bundle, MODEL_SAVE_PATH)
    print(f"\nTrained model bundle successfully saved to: {MODEL_SAVE_PATH}")

    # 8. Save metrics JSON for frontend / API access
    metrics_summary = {
        'features': FEATURE_COLUMNS,
        'feature_importances': feature_importances,
        'test_metrics': rf_metrics,
        'baseline_comparison': {
            'dummy_accuracy': dummy_metrics['accuracy'],
            'lr_accuracy': lr_metrics['accuracy'],
            'rf_accuracy': rf_metrics['accuracy'],
            'dummy_f1': dummy_metrics['f1_score_class1'],
            'lr_f1': lr_metrics['f1_score_class1'],
            'rf_f1': rf_metrics['f1_score_class1'],
        },
        'split_info': split_info,
        'confusion_matrix': {
            'tn': int(cm[0, 0]),
            'fp': int(cm[0, 1]),
            'fn': int(cm[1, 0]),
            'tp': int(cm[1, 1]),
        }
    }

    with open(METRICS_SAVE_PATH, 'w') as f:
        json.dump(metrics_summary, f, indent=4)
    print(f"Metrics JSON saved to: {METRICS_SAVE_PATH}")
    print("\nTraining and evaluation pipeline completed successfully!")


if __name__ == '__main__':
    main()
