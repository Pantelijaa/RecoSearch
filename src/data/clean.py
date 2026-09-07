"""
Cleaning pipeline for Amazon Reviews 2023 (Video_Games category).

Each function handles one specific cleaning concern, so they can be
tested/used independently in the EDA notebook, and composed together in
clean_pipeline() for the final production run.
"""

import logging
import pandas as pd
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

PROCESSED_DIR = "../../data/processed"

REVIEW_COLS = [
    "user_id",
    "parent_asin",
    "rating",
    "timestamp",
    "title",
    "text",
    "helpful_vote",
    "verified_purchase",
]

METADATA_COLS = [
    "parent_asin",
    "title",
    "description",
    "features",
    "average_rating",
    "rating_number",
    "price",
]

def select_columns(df, columns):
    df = df[columns]
    logger.info("Selected desired columns!")

    return df

def _safe_join(x):
    """
    Safely join a list/array-like field into a single string.
    Handles list, numpy array, None, and NaN gracefully -- returns "" if
    the value is empty or not usable.
    """
    if x is None:
        return ""
    try:
        if len(x) == 0:
            return "";
        return " ".join(str(item) for item in x)
    except TypeError:
        return str(x) if str(x).lower() != "nan" else ""

def verify_referential_integrity(reviews_df, metadata_df, id_col="parent_asin", drop_mismatches=True):
    """
    Verify if every item referenced in reviews_df exists in metadata_df.

    Based on EDA (Section 6): for the Video_Games category, 0 reviews reference
    missing metadata, so this is expected to be a no-op. Kept as an active check
    so the pipeline stays safe if run against a different category
    where this may not be the case.

    Parameters
    ----------
    reviews_df : pd.DataFrame
        Reviews table, must contain id_col.
    metadata_df : pd.DataFrame
        Metadata table, must contain id_col.
    id_col : str
        Column name used to join reviews and metadata (default: "parent_asin").
    drop_mismatches : bool
        If True, drop reviews whose id_col value has no matching metadata row.
        If False, only log/report the mismatch without modifying reviews_df.

    Returns
    -------
    pd.DataFrame
        reviews_df, optionally filtered to drop rows with no matching metadata.
    """
    review_items = set(reviews_df[id_col])
    metadata_items = set(metadata_df[id_col])

    missing_from_meta = review_items - metadata_items
    n_missing = len(missing_from_meta)
    pct_missing = n_missing / len(review_items) if review_items else 0.0

    if n_missing == 0:
        logger.info("Referential integrity check passed: 0 items in reviews missing from metadata.")
    else:
        logger.warning(
            f"Referential integrity issue: {n_missing} items in reviews "
            f"({pct_missing:.4%}) have no matching metadata row."
        )
        if drop_mismatches:
            before = len(reviews_df)
            reviews_df = reviews_df[~reviews_df[id_col].isin(missing_from_meta)]
            logger.info(f"Dropped mismatched reviews: {before} -> {len(reviews_df)} rows.")

    return reviews_df


def iterative_filter(df, min_user=2, min_item=5, max_iters=5):
    """
    Iteratively filter interactions so that every remaining user and item
    meets the minimum interaction threshold.

    Filtering must be iterative because removing users can drop items below
    the item threshold, and vice versa -- a single pass isn't guaranteed to
    converge to a stable dataset where BOTH thresholds hold simultaneously.

    Based on EDA (Section 3): standard cutoffs (e.g. 5+ per user) are too
    aggressive for this dataset since 75.2% of users have only 1 review.
    Chosen defaults: min_user=2, min_item=5.

    Parameters
    ----------
    df : pd.DataFrame
        Reviews/interactions table, must contain "user_id" and "parent_asin".
    min_user : int
        Minimum number of interactions a user must have to be kept.
    min_item : int
        Minimum number of interactions an item must have to be kept.
    max_iters : int
        Safety cap on filtering iterations, in case convergence is slow.

    Returns
    -------
    pd.DataFrame
        Filtered dataframe where every user and item meets both thresholds.
    """
    for i in range(max_iters):
        before = len(df)

        user_counts = df.groupby("user_id").size()
        item_counts = df.groupby("parent_asin").size()

        valid_users = user_counts[user_counts >= min_user].index
        valid_items = item_counts[item_counts >= min_item].index

        df = df[df["user_id"].isin(valid_users) & df["parent_asin"].isin(valid_items)]
        after = len(df)

        logger.info(f"Iteration {i + 1}: {before} -> {after} reviews")

        if after == before:
            logger.info(f"Converged after {i + 1} iteration(s).")
            break
    else:
        logger.warning(f"Reached max_iters={max_iters} without full convergence.")

    return df

def build_item_text(metadata_df, title_col="title", description_col="description", features_col="features"):
    """
    Build a combined `item_text` field per item using a fallback chain:
    title -> + description (if present) -> + features (if present).

    Based on EDA (Section 7): title covers ~100% of items (only 5/100k empty),
    description is empty in 37.77% of items, and features is empty in 28.81%.
    Concatenating all three (when present) maximizes usable text per item
    rather than relying on any single field.

    Parameters
    ----------
    metadata_df : pd.DataFrame
        Metadata table containing title_col, desc_col, features_col.
    title_col, description_col, features_col : str
        Column names for the three text sources.

    Returns
    -------
    pd.DataFrame
        metadata_df with a new "item_text" column added.
    """
    metadata_df = metadata_df.copy()

    titles = metadata_df[title_col].fillna("").astype(str)
    descriptions = metadata_df[description_col].apply(_safe_join)
    features = metadata_df[features_col].apply(_safe_join)

    combined = (titles + " " + descriptions + " " + features).str.strip()
    combined = combined.str.replace(r"\s+", " ", regex=True)

    metadata_df["item_text"] = combined
    n_empty = (metadata_df["item_text"].str.len() == 0).sum()
    logger.info(f"Built item_text for {len(metadata_df)} items ({n_empty} ended up empty).")

    metadata_df.drop(columns=[f"{description_col}", f"{features_col}"], inplace=True) # Keep title column for display purpose

    return metadata_df

def drop_empty_item_texts(metadata_df, text_col="item_text"):
    """
    Drop items where item_text is empty -- these items have no usable text
    from title, description, or features and cannot support text-based
    search or embedding.

    Based on EDA (Section 7): expected to affect ~2 items out of 100,000.

    Parameters
    ----------
    metadata_df : pd.DataFrame
        Metadata table, must contain text_col (see build_item_text).
    text_col : str
        Column to check for emptiness.

    Returns
    -------
    pd.DataFrame
        metadata_df with empty-text items removed.
    """
    before = len(metadata_df)
    metadata_df = metadata_df.query(f"{text_col} != ''")
    after = len(metadata_df)

    logger.info(f"Dropped {before - after} items with empty item_text ({before} -> {after}).")

    return metadata_df

def clean_pipeline():
    reviews_df = pd.read_parquet("../data/raw/Video_Games_reviews_100000.parquet")
    metadata_df = pd.read_parquet("../data/raw/Video_Games_metadata_100000.parquet")

    reviews_df = select_columns(reviews_df, REVIEW_COLS)
    metadata_df = select_columns(metadata_df, METADATA_COLS)

    reviews_df = verify_referential_integrity(reviews_df, metadata_df)
    reviews_df = iterative_filter(reviews_df)

    metadata_df = build_item_text(metadata_df)
    metadata_df = drop_empty_item_texts(metadata_df)

    # keep reviews and metadata consistent after empty-text items are dropped
    reviews_df = reviews_df[reviews_df["parent_asin"].isin(metadata_df["parent_asin"])]

    folder_path = Path(PROCESSED_DIR)
    folder_path.mkdir(parents=True, exist_ok=True)
    reviews_df.to_parquet(f"{PROCESSED_DIR}/reviews_df.parquet", index=False)
    metadata_df.to_parquet(f"{PROCESSED_DIR}/metadata_df.parquet", index=False)

if __name__ == "__main__":
    clean_pipeline()