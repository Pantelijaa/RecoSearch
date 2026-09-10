"""
Tiny matrix-factorization recommender (PyTorch), trained with BPR.

This is the simplest possible two-tower model: each "tower" is a bare
embedding lookup (user -> vector, item -> vector), and the score is their dot
product plus an item bias. It is the warm-up for the two-tower
model, which replaces the lookups with real encoders but keeps this training
setup (implicit positives + BPR + negative sampling) unchanged.
"""

import logging

import numpy as np
import torch.nn as nn
from torch import matmul
from torch.utils.data import Dataset

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def build_id_mappings(train_df, user_col="user_id", item_col="parent_asin"):
    """
    Map string user/item ids to contiguous integer indices, built from `train`.

    Embeddings are indexed lookup tables, so ids must be 0..n-1 ints. Built on
    train only: users/items absent from train get no embedding and cannot be
    scored by MF (they are handled at evaluation time via the warm-user set).

    Returns
    -------
    (dict, dict, list)
         user2idx, item2idx, and idx2item (item id per index, for inference).
    """
    users = train_df[user_col].unique()
    items = train_df[item_col].unique()

    user2idx = {user: i for i, user in enumerate(users)}
    item2idx = {item: i for i, item in enumerate(items)}
    # position == index, inverse of item2idx
    idx2items = list(items)

    logger.info(f"Built id mappings: {len(user2idx)} users, {len(item2idx)} items.")

    return user2idx, item2idx, idx2items

class BPRDataset(Dataset):
    """
    Yields (user, positive_item, negative_item) index triples for BPR.

    Positives are train interactions with rating >= positive_threshold. For
    each positive, a negative item is sampled uniformly at random from items
    the user has NOT interacted with in train (any rating), by rejection
    sampling -- so we never accidentally treat a known interaction as a
    negative.

    Parameters
    ----------
    train_df : pd.DataFrame
        Training interactions.
    user2idx, item2idx : dict
        Id -> index mappings (from build_id_mappings).
    user_col, item_col, rating_col : str
        Column names.
    positive_threshold : int
        Ratings >= this define a positive.
    seed : int
        Seed for the negative sampler (reproducibility).
    """

    def __init__(self, train_df, user2idx, item2idx,
                 user_col="user_id", item_col="parent_asin", rating_col="rating",
                 positive_threshold=4, seed=42):
        self.n_items = len(item2idx)
        self.rng = np.random.default_rng(seed)

        all_users = train_df[user_col].map(user2idx).to_numpy()
        all_items = train_df[item_col].map(item2idx).to_numpy()

        self.user_interacted = {}
        for u, i in zip(all_users, all_items):
                self.user_interacted.setdefault(u, set()).add(i)

        positives = train_df.query(f"{rating_col} >= @positive_threshold")
        self.pos_users = positives[user_col].map(user2idx).to_numpy()
        self.pos_items = positives[item_col].map(item2idx).to_numpy()

        logger.info(
            f"BPRDataset: {len(self.pos_users)} positive pairs "
            f"(rating >= {positive_threshold}) over {len(self.user_interacted)} users."
        )

    def __len__(self):
        return len(self.pos_users)

    def __getitem__(self, idx):
        u = int(self.pos_users[idx])
        i = int(self.pos_items[idx])

        seen = self.user_interacted[u]
        # Rejection sampling: draw a random item until we find one the user
        # has not interacted with. Cheap because |seen| << n_items.
        j = int(self.rng.integers(self.n_items))
        while j in seen:
            j = int(self.rng.integers(self.n_items))

        return u, i, j


class MFModel(nn.Module):
    """
    Matrix factorization: score(u, i) = user_emb[u] . item_emb[i] + item_bias[i].

    The item bias absorbs global item popularity so the embedding dot product
    is free to capture *taste* (who likes what) rather than re-learning "this
    item is popular". Embeddings are init'd small so early scores start near 0.
    """
    def __init__(self, n_users, n_items, embedding_dim=64):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, embedding_dim)
        self.item_emb = nn.Embedding(n_items, embedding_dim)
        self.item_bias = nn.Embedding(n_items, 1)

        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.item_emb.weight, std=0.01)
        nn.init.normal_(self.item_bias.weight)

    def forward(self, users, items):
        """Score given (users, items) index tensors of equal shape."""
        u = self.user_emb(users)
        v = self.item_emb(items)
        b = self.item_bias(items).squeeze(-1)

        return (u * v).sum(dim=-1) + b

    def score_all_items(self, users):
        """ Score every item for each user: (batch,) -> (batch, n_items). """
        u = self.user_emb(users)                # (B, d)
        scores = matmul(u, self.item_emb.weight.t())   # (B, n_items)
        scores = scores + self.item_bias.weight.squeeze(-1) # broadcast(n_items,)
        return scores
