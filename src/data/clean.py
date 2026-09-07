"""
Cleaning pipeline for Amazon Reviews 2023 (Video_Games category).

Each function handles one specific cleaning concern, so they can be
tested/used independently in the EDA notebook, and composed together in
clean_pipeline() for the final production run.
"""

import logging
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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

if __name__ == "__main__":
    reviews_df = pd.read_parquet("../data/raw/Video_Games_reviews_100000.parquet")
    metadata_df = pd.read_parquet("../data/raw/Video_Games_metadata_100000.parquet")

    reviews_df = verify_referential_integrity(reviews_df, metadata_df)
    reviews_df = iterative_filter(reviews_df)