import numpy as np
import pandas as pd
from server.plugins.fastcompare.algo.algorithm_base import (
    Parameter,
    ParameterType,
)

from server.plugins.fastcompare.algo.high_order_model import HigherOrderCollaborativeFiltering


class HigherOrderCFWithNovelty(HigherOrderCollaborativeFiltering):
    def __init__(self, loader, threshold, lambdaBB, lambdaCC, rho, epochs, **kwargs):
        super().__init__(loader, threshold, lambdaBB, lambdaCC, rho, epochs, **kwargs)
        print(loader.items_df.head())
        print("Initializing HigherOrderCFWithNovelty")
        self.items_df = loader.items_df
        self.average_year = self.calculate_average_year()

    def calculate_average_year(self):
        """Calculate the average year of the items."""
        return self.items_df['year'].astype(int).mean()

    def calculate_year_novelty_score(self, item_year):
        """Calculate the novelty score of an item's year based on its deviation from the average year."""
        deviation = abs(item_year - self.average_year)
        return deviation

    def calculate_novelty_score(self, item, all_items_interactions, total_users):
        """
        Calculate the novelty score of an item based on its inverse popularity (IP) and year.
        """
        item_popularity = all_items_interactions.get(item, 0) / total_users
        ip_novelty_score = -np.log2(item_popularity) if item_popularity > 0 else 0

        if item >= len(self.items_df):
            print(f"Warning: Item index {item} is out of bounds")
            year_novelty_score = 0
        else:
            item_year = int(self.items_df.iloc[item]['year'])
            year_novelty_score = self.calculate_year_novelty_score(item_year)

        # Combine the scores (weights can be adjusted based on preference)
        novelty_score = ip_novelty_score + year_novelty_score
        return novelty_score

    def predict(self, selected_items, filter_out_items, k):
        print("Starting prediction with novelty and IP")
        rat = pd.DataFrame({"item": selected_items}).set_index("item", drop=False)
        candidates = np.setdiff1d(self._all_items, rat.item.unique())
        candidates = np.setdiff1d(candidates, filter_out_items)
        if not selected_items:
            print("No items selected, returning random candidates")
            return np.random.choice(candidates, size=k, replace=False).tolist()

        indices = list(selected_items)
        user_vector = np.zeros((self._items_count,))
        for i in indices:
            user_vector[i] = 1.0

        print(f"user_vector shape: {user_vector.shape}")
        print(f"self._BB shape: {self._BB.shape}")
        print(f"self._CC shape: {self._CC.shape}")

        if user_vector.shape[0] != self._BB.shape[0]:
            raise ValueError(
                f"Incompatible shapes for user_vector and self._BB: {user_vector.shape[0]} != {self._BB.shape[0]}")
        if self._CC.shape[0] == 0:
            raise ValueError("self._CC is empty. Check the fit method for proper initialization and training.")
        if user_vector.shape[0] != self._CC.shape[1]:
            raise ValueError(
                f"Incompatible shapes for user_vector and self._CC: {user_vector.shape[0]} != {self._CC.shape[1]}")

        preds_pairwise = user_vector @ self._BB
        preds_higher_order = user_vector @ self._CC.T

        preds_higher_order_resized = np.resize(preds_higher_order, preds_pairwise.shape)

        preds = preds_pairwise + preds_higher_order_resized
        preds[user_vector.nonzero()] = -np.inf

        # Calculate novelty scores for each candidate
        total_users = self._rating_matrix.shape[0]
        all_items_interactions = {item: count for item, count in enumerate(np.sum(self._rating_matrix, axis=0))}
        novelty_scores = {item: self.calculate_novelty_score(item, all_items_interactions, total_users) for item in candidates}

        # Re-rank candidates based on the novelty score and the prediction score
        candidates_by_prob = sorted(
            ((preds[cand] * novelty_scores[cand], cand) for cand in candidates if cand < len(preds)),
            reverse=True
        )

        result = [x for _, x in candidates_by_prob][:k]

        # print(f"Prediction with novelty and IP completed: {result}")
        return result

    @classmethod
    def name(cls):
        return "HigherOrderCFWithNovelty"

    @classmethod
    def parameters(cls):
        return [
            Parameter("threshold", ParameterType.FLOAT, 100000, help="Threshold for feature pair creation"),  # Adjusted to have cca 500 pairs
            Parameter("lambdaBB", ParameterType.FLOAT, 500, help="Lambda parameter for BB regularization"),
            Parameter("lambdaCC", ParameterType.FLOAT, 2000, help="Lambda parameter for CC regularization"),
            Parameter("rho", ParameterType.FLOAT, 30000, help="Rho parameter for CC regularization"),
            Parameter("epochs", ParameterType.INT, 40, help="Number of epochs for training")
        ]
