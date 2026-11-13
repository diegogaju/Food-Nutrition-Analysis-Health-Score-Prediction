import json
import logging
import os
from pathlib import Path

import psycopg2
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

CREATE_RAW_TABLE = """
CREATE TABLE IF NOT EXISTS products_raw (
    code            TEXT PRIMARY KEY,
    product_name    TEXT,
    categories_tags TEXT,
    countries_tags  TEXT,
    nutrition_grade TEXT,
    nutriments      JSONB,
    search_category TEXT,
    loaded_at       TIMESTAMP DEFAULT NOW()
);
"""

INSERT_RAW = """
INSERT INTO products_raw (code, product_name, categories_tags, countries_tags, nutrition_grade, nutriments, search_category)
VALUES (%s, %s, %s, %s, %s, %s, %s)
ON CONFLICT (code) DO NOTHING;
"""


def get_connection():
    """Create and return a PostgreSQL connection using .env credentials."""
    try:
        conn = psycopg2.connect(
            host=os.getenv("DB_HOST", "localhost"),
            port=os.getenv("DB_PORT", 5432),
            dbname=os.getenv("DB_NAME", "food_pipeline"),
            user=os.getenv("DB_USER", "postgres"),
            password=os.getenv("DB_PASSWORD"),
        )
        logger.info("Database connection established")
        return conn
    except psycopg2.OperationalError as e:
        logger.error(f"Could not connect to database: {e}")
        raise


def load_raw(products: list[dict]) -> int:
    """
    Create products_raw table if it doesn't exist and insert products.
    Skips duplicates via ON CONFLICT DO NOTHING.
    Returns number of rows actually inserted.
    """
    conn = get_connection()
    inserted = 0
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(CREATE_RAW_TABLE)
                logger.info("products_raw table ready")

                for p in products:
                    cur.execute(INSERT_RAW, (
                        p.get("code"),
                        p.get("product_name"),
                        json.dumps(p.get("categories_tags", [])),
                        json.dumps(p.get("countries_tags", [])),
                        p.get("nutrition_grades"),
                        json.dumps(p.get("nutriments", {})),
                        p.get("_search_category"),
                    ))
                    if cur.rowcount:
                        inserted += 1

        logger.info(f"Loaded {inserted} new rows into products_raw (skipped {len(products) - inserted} duplicates)")
    finally:
        conn.close()

    return inserted


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler("logs/pipeline.log"),
            logging.StreamHandler()
        ]
    )

    # find the most recent raw file
    raw_files = sorted(Path("data").glob("raw_*.json"), reverse=True)
    if not raw_files:
        logger.error("No raw data file found in data/. Run extract.py first.")
        exit(1)

    raw_file = raw_files[0]
    logger.info(f"Loading from {raw_file}")

    with open(raw_file, encoding="utf-8") as f:
        products = json.load(f)

    logger.info(f"{len(products)} products read from file")
    load_raw(products)
