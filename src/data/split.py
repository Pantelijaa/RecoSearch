"""
Chronological train/val/test split for Amazon Reviews 2023 (Video_Games category).

Based on EDA:
    - Review volume drops sharply from May 2023 onward (data collection artifact,
      not a real activity decline) -- trim before splitting.
    - Volume and rating composition are both uneven over time, so explicit
      calendar-based cutoffs are used instead of row-count quantiles.
"""

import pandas as pd
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# Based on EDA: 2023-04-30 is the last month with a "healthy"
# review volume before the trailing collection-artifact drop-off.
TRIM_CUTOFF = pd.Timestamp("2023-05-01")

TRAIN_CUTOFF = pd.Timestamp("2021-01-01")
VAL_CUTOFF = pd.Timestamp("2022-06-01")

def trim_incomplete_tail(reviews_df, timestamp_col="timestamp", cutoff=TRIM_CUTOFF):
    """
    Drop reviews at or after `cutoff` -- based on EDA, the final months of the
    raw dataset show a sharp, implausible drop in volume (17,509 reviews in
    April 2023 down to 270 in September 2023), consistent with an incomplete
    data collection window rather than a real decline in activity. Including
    these partial months would bias whichever split they land in.

    Parameters
    ----------
    reviews_df : pd.DataFrame
        Reviews table with a timestamp column (datetime64 dtype).
    timestamp_col : str
        Name of the timestamp column.
    cutoff : pd.Timestamp
        Reviews on or after this date are dropped.

    Returns
    -------
    pd.DataFrame
        reviews_df with the incomplete tail removed.
    """
    before = len(reviews_df)
    reviews_df = reviews_df.query(f"{timestamp_col} < @cutoff")
    after = len(reviews_df)

    logger.info(f"Trimmed incomplete tail (cutoff={cutoff.date()}): {before} -> {after} reviews.")

    return reviews_df

def chronological_split(reviews_df, timestamp_col="timestamp", train_cutoff=TRAIN_CUTOFF, val_cutoff=VAL_CUTOFF):
    """
    Split reviews into train/val/test using explicit calendar cutoffs.

    Based on EDA: review volume is extremely uneven over time
    (sparse pre-2013, heavy post-2013 with seasonal spikes), so a row-count
    quantile split could land awkwardly inside a seasonal spike or the sparse
    early-years period. Calendar cutoffs give more interpretable, stable
    boundaries.

    Parameters
    ----------
    reviews_df : pd.DataFrame
        Reviews table with a timestamp column (datetime64 dtype). Should
        already have the incomplete tail trimmed (see trim_incomplete_tail).
    timestamp_col : str
        Name of the timestamp column.
    train_cutoff : pd.Timestamp
        Reviews before this date go into train.
    val_cutoff : pd.Timestamp
        Reviews from train_cutoff (inclusive) up to this date go into val;
        everything from this date onward goes into test.

    Returns
    -------
    (pd.DataFrame, pd.DataFrame, pd.DataFrame)
        train, val, test dataframes.
    """
    train = reviews_df.query(f"{timestamp_col} < @train_cutoff")
    val = reviews_df.query(f"{timestamp_col} >= @train_cutoff and {timestamp_col} < @val_cutoff ")
    test = reviews_df.query(f"{timestamp_col} >= @val_cutoff")

    total = len(reviews_df)

    logger.info(
        f"Split sizes -- train: {len(train)} ({len(train) / total:.1%}), "
        f"val: {len(val)} ({len(val) / total:.1%}), "
        f"test: {len(test)} ({len(test) / total:.1%})"
    )

    return train, val, test


def check_cold_start(train, val, test, user_col="user_id", item_col="parent_asin"):
    """
    Check for cold-start users/items in val and test -- i.e. users or items
    that never appeared in train. A model can't have learned an embedding
    for a user/item it never saw, so these need to be handled explicitly
    in evaluation (e.g. excluded from ranking metrics, or evaluated
    separately as a cold-start-specific slice) rather than silently scored
    as if they were normal cases.

    Parameters
    ----------
    train, val, test : pd.DataFrame
        Split dataframes, each containing user_col and item_col.
    user_col, item_col : str
        Column names for user and item IDs.

    Returns
    -------
    dict
        Summary of cold-start counts/rates for val and test, e.g.:
        {
            "val":  {"cold_users": int, "cold_users_pct": float,
                     "cold_items": int, "cold_items_pct": float},
            "test": {...}
        }
    """
    train_users = set(train[user_col])
    train_items = set(train[item_col])

    results = {}
    for name, split_df in [("val", val), ("test", test)]:
        split_users = set(split_df[user_col])
        split_items = set(split_df[item_col])

        cold_users = split_users - train_users
        cold_items = split_items - train_items

        n_users = len(split_users)
        n_items = len(split_items)

        results[name] = {
            "cold_users": len(cold_users),
            "cold_users_pct": len(cold_users) / n_users if n_users else 0.0,
            "cold_items": len(cold_items),
            "cold_items_pct": len(cold_items) / n_items if n_items else 0.0,
        }

        logger.info(
            f"{name}: {len(cold_users)}/{n_users} cold-start users "
            f"({results[name]['cold_users_pct']:.1%}), "
            f"{len(cold_items)}/{n_items} cold-start items "
            f"({results[name]['cold_items_pct']:.1%})"
        )

    return results


def report_split_composition(train, val, test, rating_col="rating", positive_threshold=4):
    """
    Report interaction counts and positive-rate composition per split.

    Based on EDA: rating composition is non-stationary over
    time (5-star share dipped in the mid-2000s, then rose from 2012 onward),
    so train/val/test are expected to have different positive-rate
    compositions -- this should be reported explicitly rather than assumed
    to be uniform across splits.

    Parameters
    ----------
    train, val, test : pd.DataFrame
        Split dataframes, each containing rating_col.
    rating_col : str
        Column name for the rating value.
    positive_threshold : int
        Ratings >= this value are counted as "positive" interactions.

    Returns
    -------
    pd.DataFrame
        One row per split with n_interactions and positive_rate.
    """
    rows = []
    for name, split_df in [("train", train), ("val", val), ("test", test)]:
        n = len(split_df)
        positive_rate = (split_df[rating_col] >= positive_threshold).mean() if n else float("nan")
        rows.append({"split": name, "n_interactions": n, "positive_rate": positive_rate})

    report = pd.DataFrame(rows)
    logger.info("Split composition:\n" + report.to_string(index=False))

    return report

def split_pipeline(input_df=None, input_path=None,
                    output_dir="../../data/processed",
                    timestamp_col="timestamp"):
    """
    Full split pipeline: load cleaned reviews, trim the incomplete tail,
    split chronologically into train/val/test, report cold-start and
    composition diagnostics, and save the three splits to disk.

    Parameters
    ----------
    input_path : str
        Path to the cleaned reviews parquet file (output of clean_pipeline).
    output_dir : str
        Directory to save train/val/test parquet files into.
    timestamp_col : str
        Name of the timestamp column.

    Returns
    -------
    (pd.DataFrame, pd.DataFrame, pd.DataFrame)
        train, val, test dataframes.
    """
    if input_df is None and input_path is None:
        raise ValueError("Provide either reviews_df or input_path.")

    if input_df is None:
        input_df = pd.read_parquet(input_path)

    input_df = input_df.copy()
    input_df[timestamp_col] = pd.to_datetime(input_df[timestamp_col], unit="ms")

    input_df = trim_incomplete_tail(input_df, timestamp_col=timestamp_col)
    train, val, test = chronological_split(input_df, timestamp_col=timestamp_col)

    check_cold_start(train, val, test)
    report_split_composition(train, val, test)

    folder_path = Path(output_dir)
    folder_path.mkdir(parents=True, exist_ok=True)

    train.to_parquet(f"{output_dir}/train.parquet", index=False)
    val.to_parquet(f"{output_dir}/val.parquet", index=False)
    test.to_parquet(f"{output_dir}/test.parquet", index=False)

    logger.info(f"Saved train/val/test to {output_dir}/")

    return train, val, test


if __name__ == "__main__":
    train, val, test = split_pipeline()