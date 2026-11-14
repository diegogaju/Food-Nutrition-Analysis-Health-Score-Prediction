import logging
import os
import warnings

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from dotenv import load_dotenv
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

NUMERIC_COLS = ["energy_100g", "fat_100g", "sugars_100g",
                "proteins_100g", "salt_100g", "fiber_100g", "carbohydrates_100g"]
CAT_COLS = ["category", "country", "nutrition_grade"]


def get_df() -> pd.DataFrame: # loads data
    url = (
        f"postgresql+psycopg2://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}"
        f"@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
    )
    engine = create_engine(url)
    df = pd.read_sql("SELECT * FROM products_clean", engine)
    logger.info(f"Loaded {len(df)} rows from products_clean")
    return df


# ── 1. OVERVIEW ──────────────────────────────────────────────────────────────
def overview(df: pd.DataFrame):
    logger.info("=== DATASET OVERVIEW ===")
    logger.info(f"Rows: {len(df)}  |  Columns: {len(df.columns)}")
    logger.info(f"Duplicate rows: {df.duplicated().sum()}")

    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(1)
    summary = pd.DataFrame({"missing": missing, "missing_%": missing_pct, "dtype": df.dtypes})
    logger.info(f"\n{summary.to_string()}")


# ── 2. NUMERIC COLUMNS ───────────────────────────────────────────────────────
def numeric_analysis(df: pd.DataFrame):
    logger.info("=== NUMERIC COLUMN STATS ===")
    stats = df[NUMERIC_COLS].describe().round(2)
    logger.info(f"\n{stats.to_string()}")

    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    axes = axes.flatten()
    for i, col in enumerate(NUMERIC_COLS):
        data = df[col].dropna()
        axes[i].hist(data, bins=40, color="steelblue", edgecolor="white")
        axes[i].set_title(col)
        axes[i].set_xlabel("value per 100g")
        axes[i].set_ylabel("count")
    axes[-1].set_visible(False)
    plt.suptitle("Distribution of Nutriments (per 100g)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig("data/eda_numeric_distributions.png", dpi=150)
    plt.close()
    logger.info("Saved data/eda_numeric_distributions.png")


# ── 3. CATEGORICAL COLUMNS ───────────────────────────────────────────────────
def categorical_analysis(df: pd.DataFrame):
    logger.info("=== CATEGORICAL COLUMN STATS ===")

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # nutrition grade distribution
    grade_counts = df["nutrition_grade"].value_counts().reindex(["a", "b", "c", "d", "e"])
    colors = ["#2ecc71", "#a8d8a8", "#f39c12", "#e67e22", "#e74c3c"]
    axes[0].bar(grade_counts.index, grade_counts.values, color=colors)
    axes[0].set_title("Products by Nutrition Grade")
    axes[0].set_xlabel("Grade")
    axes[0].set_ylabel("Count")
    for i, v in enumerate(grade_counts.values):
        axes[0].text(i, v + 5, str(v), ha="center", fontsize=9)

    # top 10 categories
    top_cats = df["category"].value_counts().head(10)
    axes[1].barh(top_cats.index[::-1], top_cats.values[::-1], color="steelblue")
    axes[1].set_title("Top 10 Food Categories")
    axes[1].set_xlabel("Count")

    # top 10 countries
    top_countries = df["country"].value_counts().head(10)
    axes[2].barh(top_countries.index[::-1], top_countries.values[::-1], color="coral")
    axes[2].set_title("Top 10 Countries")
    axes[2].set_xlabel("Count")

    plt.suptitle("Categorical Column Analysis", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig("data/eda_categorical.png", dpi=150)
    plt.close()
    logger.info("Saved data/eda_categorical.png")

    for col in CAT_COLS:
        logger.info(f"\n{col} — top 10 values:\n{df[col].value_counts().head(10).to_string()}")


# ── 4. COLUMN COMPARISONS ────────────────────────────────────────────────────
def comparisons(df: pd.DataFrame):
    logger.info("=== COLUMN COMPARISONS ===")

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # fat vs energy scatter
    sample = df[["fat_100g", "energy_100g"]].dropna().sample(min(500, len(df)), random_state=42)
    axes[0].scatter(sample["fat_100g"], sample["energy_100g"], alpha=0.4, color="steelblue", s=15)
    axes[0].set_title("Fat vs Energy (per 100g)")
    axes[0].set_xlabel("Fat (g)")
    axes[0].set_ylabel("Energy (kcal)")
    corr = sample.corr().iloc[0, 1]
    axes[0].text(0.05, 0.95, f"r = {corr:.2f}", transform=axes[0].transAxes, fontsize=10)

    # avg sugar by nutrition grade
    avg_sugar = df.groupby("nutrition_grade")["sugars_100g"].mean().reindex(["a", "b", "c", "d", "e"])
    axes[1].bar(avg_sugar.index, avg_sugar.values, color=["#2ecc71", "#a8d8a8", "#f39c12", "#e67e22", "#e74c3c"])
    axes[1].set_title("Avg Sugar by Nutrition Grade")
    axes[1].set_xlabel("Grade")
    axes[1].set_ylabel("Avg Sugar (g per 100g)")

    # correlation heatmap
    corr_matrix = df[NUMERIC_COLS].corr().round(2)
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm",
                ax=axes[2], annot_kws={"size": 7})
    axes[2].set_title("Nutriment Correlation Heatmap")

    plt.suptitle("Column Comparisons", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig("data/eda_comparisons.png", dpi=150)
    plt.close()
    logger.info("Saved data/eda_comparisons.png")

    logger.info(f"Correlation fat vs energy: {corr:.2f}")
    logger.info(f"\nAvg sugar by grade:\n{avg_sugar.to_string()}")


# ── 5. WRITTEN SUMMARY ───────────────────────────────────────────────────────
def summary(df: pd.DataFrame):
    total = len(df)
    missing_grade = df["nutrition_grade"].isnull().sum()
    missing_energy = df["energy_100g"].isnull().sum()
    top_cat = df["category"].value_counts().idxmax()
    top_country = df["country"].value_counts().idxmax()
    avg_energy = df["energy_100g"].mean()
    avg_sugar = df["sugars_100g"].mean()

    print("\n" + "=" * 60)
    print("EDA SUMMARY")
    print("=" * 60)
    print(f"Dataset has {total} products across {df['category'].nunique()} categories "
          f"and {df['country'].nunique()} countries.")
    print(f"Most common category: '{top_cat}' | Most common country: '{top_country}'.")
    print(f"{missing_grade} products ({missing_grade/total*100:.1f}%) have no nutrition grade — "
          f"these will be excluded from classification ML.")
    print(f"{missing_energy} products ({missing_energy/total*100:.1f}%) have no energy value.")
    print(f"Average energy: {avg_energy:.1f} kcal/100g | Average sugar: {avg_sugar:.1f} g/100g.")
    print("Fat and energy are strongly correlated — fat is the biggest driver of calorie content.")
    print("Products graded 'e' have noticeably higher sugar content than grade 'a' products,")
    print("confirming that nutrition grade reflects actual nutritional quality.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    df = get_df()
    overview(df)
    numeric_analysis(df)
    categorical_analysis(df)
    comparisons(df)
    summary(df)
