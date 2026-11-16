import logging
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

Path("logs").mkdir(exist_ok=True)
Path("data").mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler("logs/pipeline.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def run():
    start = datetime.now()
    logger.info("=" * 60)
    logger.info("Pipeline started")

    # ── STEP 1: EXTRACT ───────────────────────────────────────────
    logger.info("STEP 1/3 — Extract: fetching products from Open Food Facts API")
    try:
        from extract import extract, save_raw
        products = extract()
        if not products:
            logger.error("Extract returned 0 products — aborting pipeline")
            sys.exit(1)
        raw_path = save_raw(products)
        logger.info(f"STEP 1/3 — Extract complete: {len(products)} products saved to {raw_path}")
    except Exception as e:
        logger.error(f"STEP 1/3 — Extract failed: {e}")
        sys.exit(1)

    # ── STEP 2: LOAD RAW ──────────────────────────────────────────
    logger.info("STEP 2/3 — Load: inserting raw data into PostgreSQL")
    try:
        from load import get_connection, load_raw
        inserted = load_raw(products)
        logger.info(f"STEP 2/3 — Load complete: {inserted} new rows in products_raw")
    except Exception as e:
        logger.error(f"STEP 2/3 — Load failed: {e}")
        sys.exit(1)

    # ── STEP 3: TRANSFORM ─────────────────────────────────────────
    logger.info("STEP 3/3 — Transform: cleaning data into products_clean")
    try:
        from transform import transform_and_load
        conn = get_connection()
        try:
            stats = transform_and_load(conn)
        finally:
            conn.close()
        logger.info(
            f"STEP 3/3 — Transform complete: "
            f"{stats['inserted']} rows inserted, "
            f"{stats['dropped_no_code']} dropped, "
            f"{stats['skipped_duplicate']} skipped"
        )
    except Exception as e:
        logger.error(f"STEP 3/3 — Transform failed: {e}")
        sys.exit(1)

    elapsed = (datetime.now() - start).seconds
    logger.info(f"Pipeline completed successfully in {elapsed}s")
    logger.info("=" * 60)


if __name__ == "__main__":
    run()
