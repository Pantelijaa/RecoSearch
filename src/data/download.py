from datasets import load_dataset
from pathlib import Path
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

raw_path = "../../data/raw/"
CATEGORY = "Video_Games"
SAMPLE_SIZE = 100_000

def sample():
    reviews = load_dataset("McAuley-Lab/Amazon-Reviews-2023",
                       f"raw_review_{CATEGORY}",
                            split="full",
                           trust_remote_code=True # Required by McAuley-Lab/Amazon-Reviews-2023
                       )

    metadata = load_dataset("McAuley-Lab/Amazon-Reviews-2023",
                        f"raw_meta_{CATEGORY}",
                        split="full",
                        trust_remote_code=True
                        )
    if SAMPLE_SIZE > len(metadata):
        raise ValueError(f"Sample size ({SAMPLE_SIZE}) is biggerr than the number of metadata items ({len(metadata)})")

    # sample items
    metadata_sample = metadata.shuffle(seed=42).select(range(SAMPLE_SIZE))
    metadata_sample_df = metadata_sample.to_pandas()
    sampled_item_ids = set(metadata_sample_df["parent_asin"])

    # filter review for items
    reviews_df = reviews.to_pandas()
    reviews_sample_df = reviews_df[reviews_df["parent_asin"].isin(sampled_item_ids)]

    return reviews_sample_df, metadata_sample_df

def download(reviews, metadata):

    folder_path = Path(raw_path)
    folder_path.mkdir(parents=True, exist_ok=True)

    reviews.to_parquet(f"{raw_path}{CATEGORY}_reviews_{SAMPLE_SIZE}.parquet")
    metadata.to_parquet(f"{raw_path}{CATEGORY}_metadata_{SAMPLE_SIZE}.parquet")

    logger.info(f"Saved {len(reviews)} reviews and {len(metadata)} items")


if __name__ == "__main__":
    sampled_reviews, sampled_metadata = sample()
    download(sampled_reviews, sampled_metadata)

