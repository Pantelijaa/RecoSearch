"""
Top-K ranking metrics for recommendation and search evaluation.

The evaluation contract, for both recommenders and search:
    - The model produces a *ranked* list of item ids per user/query.
    - `relevant` is the set of ground-truth item ids that count as a hit
      (for recs: the user's held-out positive interactions; for search:
      the judged-relevant items for a query).
    - Metrics reward putting relevant items into the top K, and (for NDCG)
      putting them near the top of the top K.

Design decisions:
    - Binary relevance by default. `relevant` is a *set*, so an item is
      either a hit or not. This matches the positive_threshold>=4 convention
      used in split.py and the positive/negative framing used for MF training.
    - Metrics are computed per user, then macro-averaged across users
      (every user is one vote, regardless of how many interactions they have).
    - Users with an empty `relevant` set are unscoreable and are *skipped*,
      not counted as zero -- scoring them as zero would silently punish the
      model for users it was never given a fair chance on.
"""

import logging
import math

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def precision_at_k(recommended, relevant, k) -> float:
    """
    Fraction of the top-k recommended items that are relevant.

    precision@k = |top_k(recommended) ∩ relevant| / k

    The denominator is k (not len(recommended)), so a model that returns
    fewer than k items is penalized for the empty slots -- it was asked for
    k recommendations and did not fill them.

    Parameters
    ----------
    recommended : sequence
        Item ids in rank order (best first). Only the first k are used.
    relevant : set
        Ground-truth relevant item ids for this user/query.
    k : int
        Cutoff rank (must be positive).

    Returns
    -------
    float
        Precision in [0, 1].
    """
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}.")

    relevant_set = set(relevant)
    top_k = recommended[:k]
    hits = sum(1 for item in top_k if item in relevant_set)

    return hits / k

def recall_at_k(recommended, relevant, k) -> float:
    """
    Fraction of the user's relevant items that appear in the top-k list.

    recall@k = |top_k(recommended) ∩ relevant| / |relevant|

    Note: when |relevant| > k, recall is capped below 1.0 by construction
    (you cannot fit >k relevant items into k slots). This is expected and is
    why recall@k is always read alongside a k chosen for the use case.

    Parameters
    ----------
    recommended : sequence
        Item ids in rank order (best first). Only the first k are used.
    relevant : set
        Ground-truth relevant item ids for this user/query. Must be non-empty
        (callers should skip users with no relevant items -- see
        evaluate_recommendations).
    k : int
        Cutoff rank (must be positive).

    Returns
    -------
    float
        Recall in [0, 1].
    """
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}.")

    relevant = set(relevant)
    if not relevant:
        raise ValueError(f"recall_at_k is undefined for an empty relevant set.")

    top_k = recommended[:k]
    hits = sum(1 for item in top_k if item in relevant)

    return hits / len(relevant)

def _dcg_at_k(recommended, relevant, k) -> float:
    """
    Discounted Cumulative Gain at k, with binary gains.

    Each relevant item in the top k contributes 1 / log2(rank + 1), where
    rank is 1-based. A hit at rank 1 contributes 1 / log2(2) = 1.0; a hit at
    rank 2 contributes 1 / log2(3) ≈ 0.63; and so on -- later hits are
    discounted, encoding "users read from the top".

    Parameters
    ----------
    recommended : sequence
        Item ids in rank order (best first). Only the first k are used.
    relevant : set
        Ground-truth relevant item ids.
    k : int
        Cutoff rank.

    Returns
    -------
    float
        DCG (unnormalized; compare only via ndcg_at_k).
    """

    relevant = set(relevant)
    top_k = recommended[:k]

    dcg = 0.0
    for i, item in enumerate(top_k):
        if item in relevant:
            dcg += 1.0 / math.log2(i + 2) # equal to rank + 1
    return dcg

def ndcg_at_k(recommended, relevant, k) -> float:
    """
    Normalized DCG at k, with binary gains, in [0, 1].

    ndcg@k = dcg@k / idcg@k, where idcg@k is the DCG of the *ideal* ranking
    (all relevant items packed into the top positions). Normalizing makes the
    metric comparable across users who have different numbers of relevant
    items: a user with 2 relevant items can still score 1.0 if both are
    ranked first and second.

    The ideal ranking has min(|relevant|, k) hits, so idcg sums the discount
    over exactly that many top positions.

    Parameters
    ----------
    recommended : sequence
        Item ids in rank order (best first). Only the first k are used.
    relevant : set
        Ground-truth relevant item ids. Must be non-empty.
    k : int
        Cutoff rank.

    Returns
    -------
    float
        NDCG in [0, 1].
    """
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}.")

    relevant_set = set(relevant)
    if not relevant_set:
        raise ValueError(f"ndcg_at_k is undefined for an empty relevant set.")

    dcg = _dcg_at_k(recommended, relevant, k)
    n_ideal_hits = min(len(relevant), k)

    idcg = sum(1.0 / math.log2(i + 2) for i in range(n_ideal_hits))

    return dcg / idcg

def hit_rate_at_k(recommended, relevant, k) -> float:
    """
    1.0 if at least one relevant item is in the top k, else 0.0.

    Also called Hit Ratio@k. Macro-averaged over users, it answers "for what
    fraction of users did we surface *anything* useful in the top k?". Useful
    as a sanity floor for early baselines.

    Parameters
    ----------
    recommended : sequence
        Item ids in rank order (best first). Only the first k are used.
    relevant : set
        Ground-truth relevant item ids.
    k : int
        Cutoff rank.

    Returns
    -------
    float
        0.0 or 1.0.
    """
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}.")
    relevant_set = set(relevant)
    top_k = recommended[:k]

    return 1.0 if any(item in relevant_set for item in top_k) else 0.0

# Registry of the per-user metric functions, so the aggregator can loop over
# them by name rather than hard-coding each one.
METRIC_FNS = {
    "precision": precision_at_k,
    "recall": recall_at_k,
    "ndcg": ndcg_at_k,
    "hit_rate": hit_rate_at_k,
}

def evaluate_recommendations(recommendations, ground_truth, k_values=(5, 10, 20),
                             metrics=("precision", "recall", "ndcg", "hit_rate")):
    """
    Macro-average the ranking metrics over all evaluable users.

    For every user present in `ground_truth` with a non-empty relevant set,
    each metric is computed at each k and then averaged across those users.
    Users with an empty relevant set, or with no entry in `recommendations`,
    are handled explicitly (see below) rather than silently distorting the
    average.

    Parameters
    ----------
    recommendations : dict[user_id, sequence]
        Ranked item-id list per user (best first). A user missing from this
        dict is treated as having produced an empty list -- i.e. scored 0 on
        every metric -- because failing to recommend for a user we *can*
        evaluate is a real model failure, not an unscoreable case.
    ground_truth : dict[user_id, set]
        Relevant item-id set per user. Users with an empty set are skipped.
    k_values : iterable of int
        Cutoff ranks to report.
    metrics : iterable of str
        Which metrics to compute; keys of METRIC_FNS.

    Returns
    -------
    pd.DataFrame
        Indexed by k (one row per cutoff), one column per metric
        ("precision", "recall", "ndcg", "hit_rate"), plus an "n_users" column
        recording how many users the averages were taken over.
    """
    evaluable_user = [user for user, rel in ground_truth.items() if rel]
    n_skipped = len(ground_truth) - len(evaluable_user)
    if n_skipped:
        logger.info(f"Skipped {n_skipped} users with no relevant (positive) held-out items.")

    n_missing = sum(1 for u in evaluable_user if u not in recommendations)
    if n_missing:
        logger.info(f"{n_missing} evaluable users have no recommendations (scored 0 on all metrics).")

    rows = []
    for k in k_values:
        row = {"k": k}
        for metric in metrics:
            func = METRIC_FNS[metric]
            scores = [
                func(recommendations.get(user, []), ground_truth[user], k) for user in evaluable_user
            ]
            row[metric] = sum(scores) / len(scores) if scores else float("nan")
        rows.append(row)

    report = pd.DataFrame(rows).set_index("k")
    report["n_users"] = len(evaluable_user)

    logger.info("Ranking metrics:\n" + report.to_string())

    return report

def build_ground_truth(reviews_df, user_col="user_id", item_col="parent_asin",
                       rating_col="rating", positive_threshold=4):
    """
    Build the {user_id -> set of relevant item_ids} ground-truth mapping from
    a held-out split.

    An item is relevant for a user if the user interacted with it in this
    split AND rated it >= positive_threshold. This matches the positive/
    negative convention used in split.py's composition report and (later) in
    MF negative sampling, so evaluation and training agree on what "positive"
    means.

    Parameters
    ----------
    reviews_df : pd.DataFrame
        A held-out split (val or test) with user, item, and rating columns.
    user_col, item_col, rating_col : str
        Column names.
    positive_threshold : int
        Ratings >= this value are treated as relevant.

    Returns
    -------
    dict[user_id, set]
        Relevant item ids per user (users with only sub-threshold ratings in
        this split map to an empty set and will be skipped by the evaluator).
    """

    positives = reviews_df.query(f"{rating_col} >= @positive_threshold")
    ground_truth = (
        positives.groupby(user_col)[item_col]
        .agg(lambda items: set(items))
        .to_dict()
    )

    logger.info(
        f"Built ground truth for {len(ground_truth)} users "
        f"from {len(positives)} positive interactions "
        f"(rating >= {positive_threshold})."
    )

    return ground_truth