"""
Non-personalized baselines for the recommendation task.

These are the "dumbest thing that works" reference points that every later
model must beat to justify its complexity.

Popularity baseline:
    - Rank items globally by how often they were interacted with in TRAIN.
    - Recommend that same ranked list to every user, after removing items the
      user already saw in train.
    - It is non-personalized, so its only per-user variation comes from the
      seen-item filter. Despite that, it is typically a strong baseline,
      because real interaction data is heavily concentrated on a few popular
      items (popularity bias).
"""

import logging

import pandas as pd

from src.evaluation.metrics import build_ground_truth, evaluate_recommendations

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

def compute_item_popularity(train_df, item_col="parent_asin", rating_col="rating", positive_threshold=4, by="positive_count"):
    """
    Rank items by popularity in the training split.

    Parameters
    ----------
    train_df : pd.DataFrame
        Training interactions (must contain item_col, and rating_col if
        by="positive_count").
    item_col, rating_col : str
        Column names.
    positive_threshold : int
        Ratings >= this count as positive (used only when by="positive_count").
    by : {"positive_count", "count"}
        "positive_count": rank by number of positive (rating >= threshold)
        interactions -- matches the evaluation's notion of a relevant hit.
        "count": rank by raw interaction volume (the classic textbook
        baseline), regardless of rating.

    Returns
    -------
    (list, pd.Series)
        ranked_items: item ids, most popular first.
        counts: the popularity score per item (indexed by item id, descending).
    """

    if by == "positive_count":
        scored = train_df.query(f"{rating_col} >= positive_count")
    elif by == "count":
        scored = train_df
    else:
        raise ValueError(f"Unknown popularity mode by={by!r}; use 'positive_count' or 'count'.")

    counts = scored[item_col].value_counts()
    ranked_items = counts.index.tolist()

    logger.info(
        f"Computed item popularity by={by}: {len(ranked_items)} items ranked "
        f"from {len(scored)} interactions. Top item has {int(counts.iloc[0])} hits."
    )

    return ranked_items, counts