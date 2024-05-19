from abc import ABC
import numpy as np

from server.plugins.fastcompare.algo.algorithm_base import ParameterType, Parameter
from server.plugins.fastcompare.algo.high_order_model import HigherOrderEASE


class NoveltyBasedEASE(HigherOrderEASE, ABC):
    """Implementation of Novelty-Based EASE algorithm for EasyStudy using ADMM optimization
    Incorporates novelty metrics to improve recommendation diversity and user satisfaction
    """

    def __init__(self, loader, positive_threshold, l2, l2_C, rho, m, latent_dim=10, novelty_weight=0.5, **kwargs):
        super().__init__(loader, positive_threshold, l2, l2_C, rho, m, latent_dim, **kwargs)
        self._novelty_weight = novelty_weight

    def fit(self):
        print("Training Novelty-Based EASE model...")
        super().fit()
        print("Novelty-Based EASE model trained successfully.")

    def novelty_score(self, item_id):
        """Calculate the novelty score for an item based on its popularity"""
        user_count = len(self._rating_matrix)
        item_popularity = np.sum(self._rating_matrix[:, item_id] > 0)
        novelty = 1 - (item_popularity / user_count)
        return novelty

    def predict(self, selected_items, filter_out_items, k):
        print("Generating predictions using Novelty-Based EASE model...")

        # Create user interaction vector (2-dimensional)
        user_vector = np.zeros((1, self._items_count), dtype=np.float32)
        for item_id in selected_items:
            user_vector[0, item_id] = 1.0  # Set the corresponding element to 1

        # Pairwise predictions
        pairwise_preds = user_vector @ self._weights

        # Higher-order predictions
        Z_u = (user_vector @ self._M.T >= 2).astype(np.float32)
        higher_order_preds = Z_u @ self._C

        # Combine predictions
        preds = pairwise_preds + higher_order_preds

        # Calculate novelty scores and incorporate them into the predictions
        novelty_scores = np.array([self.novelty_score(i) for i in range(self._items_count)])
        preds = preds + self._novelty_weight * novelty_scores

        # Filter out selected and excluded items
        candidate_items = np.setdiff1d(self._all_items, selected_items)
        candidate_items = np.setdiff1d(candidate_items, filter_out_items)

        # Get predicted scores for candidate items
        candidate_scores = preds.numpy()[0, candidate_items]

        # Get indices of top-k items
        top_k_indices = np.argsort(candidate_scores)[::-1][:k]

        # Return top-k item IDs
        return candidate_items[top_k_indices]

    @classmethod
    def name(cls):
        return "Novelty-Based High-Order EASE"

    @classmethod
    def parameters(cls):
        base_params = super().parameters()
        novelty_params = [
            Parameter(
                "novelty_weight",
                ParameterType.FLOAT,
                0.5,
                help="Weight of the novelty score in the final prediction",
            ),
        ]
        return base_params + novelty_params
