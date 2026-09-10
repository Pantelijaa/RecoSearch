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

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
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
        scored = train_df.query(f"{rating_col} >= @positive_threshold")
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

def build_seen_items(train_df, user_col="user_id", item_col="parent_asin") -> dict[str, set]:
    """
    Map each training user to the set of items they interacted with in train.

    Used to avoid recommending items a user has already seen.

    Returns
    -------
    dict[user_id, set]
    """
    seen = train_df.groupby(user_col)[item_col].agg(list).to_dict()
    logger.info(f"Built seen-item set for {len(seen)} training users.")
    return seen

def recommend_popular(users, ranked_items, seen_items, k=20) -> dict:
    """
    Produce top-k popularity recommendations for each user, skipping items the
    user already saw in train.

    We scan the global ranked list and stop as soon as k unseen items are
    collected, so the per-user cost is ~k (plus the user's usually-small seen
    set), not the full catalog.

    Parameters
    ----------
    users : iterable of user_id
        Users to generate recommendations for (e.g. the val ground-truth users).
    ranked_items : list
        Item ids most-popular-first (from compute_item_popularity).
    seen_items : dict[user_id, set]
        Items each user already interacted with in train.
    k : int
        Number of items to recommend per user.

    Returns
    -------
    dict[user_id, list]
        Ranked recommendation list (length <= k) per user.
    """
    recommendations = {}
    for user in users:
        seen = seen_items.get(user, set())
        recs = []
        for item in ranked_items:
            if item in seen:
                continue
            recs.append(item)
            if len(recs) >= k:
                break
        recommendations[user] = recs

    return recommendations

def popularity_baseline_pipeline(train_path="data/processed/train.parquet",
                                 val_path="data/processed/val.parquet",
                                 k_values=(5, 10, 20), by="positive_count"):
    """
    Full popularity-baseline run: fit popularity on train, recommend for the
    val ground-truth users, and evaluate with the ranking metrics.

    Returns
    -------
    (pd.DataFrame, dict)
        report: the ranking-metrics table.
        recommendations: the per-user recommendation lists (kept for
        inspection / later comparison against other models).
    """
    train = pd.read_parquet(train_path)
    val = pd.read_parquet(val_path)

    ground_truth = build_ground_truth(val)

    ranked_items, _  = compute_item_popularity(train, by=by)
    seen_items = build_seen_items(train)

    users = list(ground_truth.keys())
    recommendations = recommend_popular(users, ranked_items, seen_items, k=max(k_values))

    logger.info(f"Popularity baseline (by={by}) on val:")
    report = evaluate_recommendations(recommendations, ground_truth, k_values=k_values)

    return report, recommendations
