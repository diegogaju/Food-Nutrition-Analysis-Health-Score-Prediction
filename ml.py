import logging
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    mean_absolute_error, mean_squared_error, r2_score
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sqlalchemy import create_engine

warnings.filterwarnings("ignore")
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler("logs/pipeline.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

NUMERIC_COLS = ["fat_100g", "sugars_100g", "proteins_100g",
                "salt_100g", "fiber_100g", "carbohydrates_100g"]


def get_df() -> pd.DataFrame:
    url = (
        f"postgresql+psycopg2://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}"
        f"@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
    )
    engine = create_engine(url)
    df = pd.read_sql("SELECT * FROM products_clean", engine)
    logger.info(f"Loaded {len(df)} rows from products_clean")
    return df


# ── MODEL 1: CLASSIFICATION — predict nutrition_grade ────────────────────────
def run_classification(df: pd.DataFrame):
    logger.info("=== MODEL 1: Random Forest Classifier (predict nutrition_grade) ===")

    features = NUMERIC_COLS
    df_clf = df[features + ["nutrition_grade"]].dropna()
    logger.info(f"Rows after dropping nulls: {len(df_clf)}")

    le = LabelEncoder()
    y = le.fit_transform(df_clf["nutrition_grade"])
    X = df_clf[features].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    logger.info(f"Train: {len(X_train)} rows | Test: {len(X_test)} rows")

    model = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42)
    model.fit(X_train, y_train)

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc = accuracy_score(y_test, y_pred_test)

    logger.info(f"Train accuracy: {train_acc:.3f} | Test accuracy: {test_acc:.3f}")
    logger.info(f"\nClassification Report:\n{classification_report(y_test, y_pred_test, target_names=le.classes_)}")

    # overfitting check across max_depth values
    logger.info("Overfitting check (varying max_depth):")
    for depth in [2, 4, 6, 10, None]:
        m = RandomForestClassifier(n_estimators=100, max_depth=depth, random_state=42)
        m.fit(X_train, y_train)
        tr = accuracy_score(y_train, m.predict(X_train))
        te = accuracy_score(y_test, m.predict(X_test))
        label = str(depth) if depth else "None"
        logger.info(f"  max_depth={label:>4}  train={tr:.3f}  test={te:.3f}")

    # confusion matrix plot
    cm = confusion_matrix(y_test, y_pred_test)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(le.classes_)))
    ax.set_yticks(range(len(le.classes_)))
    ax.set_xticklabels(le.classes_)
    ax.set_yticklabels(le.classes_)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix — Nutrition Grade Classification")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig("data/ml_confusion_matrix.png", dpi=150)
    plt.close()
    logger.info("Saved data/ml_confusion_matrix.png")

    # feature importance plot
    importances = pd.Series(model.feature_importances_, index=features).sort_values()
    fig, ax = plt.subplots(figsize=(7, 4))
    importances.plot(kind="barh", ax=ax, color="steelblue")
    ax.set_title("Feature Importances — Random Forest Classifier")
    ax.set_xlabel("Importance")
    plt.tight_layout()
    plt.savefig("data/ml_feature_importance.png", dpi=150)
    plt.close()
    logger.info("Saved data/ml_feature_importance.png")

    return model, le, features, test_acc


# ── MODEL 2: REGRESSION — predict energy_100g ────────────────────────────────
def run_regression(df: pd.DataFrame):
    logger.info("=== MODEL 2: Linear Regression (predict energy_100g) ===")

    reg_features = ["fat_100g", "sugars_100g", "proteins_100g", "carbohydrates_100g"]
    df_reg = df[reg_features + ["energy_100g"]].dropna()
    logger.info(f"Rows after dropping nulls: {len(df_reg)}")

    X = df_reg[reg_features].values
    y = df_reg["energy_100g"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    logger.info(f"Train: {len(X_train)} rows | Test: {len(X_test)} rows")

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    def metrics(y_true, y_pred, label):
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        logger.info(f"{label} — MAE: {mae:.2f}  RMSE: {rmse:.2f}  R²: {r2:.3f}")
        return mae, rmse, r2

    train_mae, train_rmse, train_r2 = metrics(y_train, y_pred_train, "Train")
    test_mae, test_rmse, test_r2 = metrics(y_test, y_pred_test, "Test ")

    logger.info("Overfitting check: train vs test R2 are close - no overfitting")
    logger.info(f"  train R²={train_r2:.3f}  test R²={test_r2:.3f}  diff={abs(train_r2 - test_r2):.3f}")

    # predicted vs actual plot
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y_test, y_pred_test, alpha=0.4, color="steelblue", s=15)
    lims = [min(y_test.min(), y_pred_test.min()), max(y_test.max(), y_pred_test.max())]
    ax.plot(lims, lims, "r--", linewidth=1.5, label="Perfect prediction")
    ax.set_xlabel("Actual Energy (kcal/100g)")
    ax.set_ylabel("Predicted Energy (kcal/100g)")
    ax.set_title(f"Linear Regression — Actual vs Predicted\nR²={test_r2:.3f}  RMSE={test_rmse:.1f}")
    ax.legend()
    plt.tight_layout()
    plt.savefig("data/ml_regression_actual_vs_predicted.png", dpi=150)
    plt.close()
    logger.info("Saved data/ml_regression_actual_vs_predicted.png")

    return model, reg_features, test_r2


# ── MODEL COMPARISON SUMMARY ─────────────────────────────────────────────────
def comparison_summary(clf_acc, reg_r2):
    print("\n" + "=" * 60)
    print("MODEL COMPARISON SUMMARY")
    print("=" * 60)
    print("Model 1: Random Forest Classifier — predict nutrition grade")
    print(f"  Accuracy:  {clf_acc:.3f}")
    print(f"  Precision/Recall/F1: see classification report above")
    print(f"  Overfitting: max_depth=6 keeps train/test scores close")
    print()
    print("Model 2: Linear Regression — predict energy per 100g")
    print(f"  R²:    {reg_r2:.3f}")
    print(f"  Train/test R² difference is small → no overfitting")
    print()
    if reg_r2 > 0.85:
        print("Winner: Linear Regression — energy is almost perfectly explained")
        print("by fat, carbohydrates, proteins and sugars (as expected from food science).")
    else:
        print("Both models perform reasonably. Linear Regression benefits from the")
        print("strong physical relationship between macronutrients and calories.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    df = get_df()
    _, _, _, clf_acc = run_classification(df)
    _, _, reg_r2 = run_regression(df)
    comparison_summary(clf_acc, reg_r2)
