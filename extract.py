import requests
import json
import logging
import time
from datetime import date
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler("logs/pipeline.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

BASE_URL = "https://world.openfoodfacts.org/api/v2/search"
FIELDS = "code,product_name,categories_tags,countries_tags,nutrition_grades,nutriments"
CATEGORIES = ["beverages", "snacks", "dairy", "cereals", "meats"]
PAGE_SIZE = 100
PAGES_PER_CATEGORY = 5
MAX_RETRIES = 3
RAW_DIR = Path("data")


def fetch_page(category: str, page: int) -> dict | None:
    """Fetch one page of products for a category. Retries up to MAX_RETRIES times."""
    params = {
        "categories_tags": category,
        "fields": FIELDS,
        "page_size": PAGE_SIZE,
        "page": page,
    }
    headers = {
        "User-Agent": "FoodPipelineProject/1.0 (bigdata-course; contact@student.edu)"
    }
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.get(BASE_URL, params=params, headers=headers, timeout=30) 
            response.raise_for_status()
            data = response.json()
            count = len(data.get("products", []))
            logger.info(f"[{category}] page {page} attempt {attempt} — {count} products received")
            return data
        except requests.RequestException as e:
            logger.warning(f"[{category}] page {page} attempt {attempt} failed: {e}")
            if attempt < MAX_RETRIES:
                time.sleep(2 ** attempt)  # exponential backoff: 2s, 4s
    logger.error(f"[{category}] page {page} — all {MAX_RETRIES} attempts failed, skipping")
    return None


def extract() -> list[dict]:
    """Fetch all products across all categories and return as a flat list."""
    all_products = []

    for category in CATEGORIES:
        logger.info(f"Starting extract for category: {category}")
        for page in range(1, PAGES_PER_CATEGORY + 1):
            data = fetch_page(category, page)
            if data is None:
                continue
            products = data.get("products", [])
            if not products:
                logger.info(f"[{category}] no more products at page {page}, stopping early")
                break
            # tag each product with the category we searched for
            for p in products:
                p["_search_category"] = category
            all_products.extend(products)

    logger.info(f"Extract complete — {len(all_products)} total products fetched")
    return all_products


def save_raw(products: list[dict]) -> Path:
    """Save raw products to a dated JSON file in data/."""
    RAW_DIR.mkdir(exist_ok=True)
    path = RAW_DIR / f"raw_{date.today()}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(products, f, ensure_ascii=False, indent=2)
    logger.info(f"Raw data saved to {path} ({len(products)} products)")
    return path


if __name__ == "__main__":
    products = extract()
    save_raw(products)
