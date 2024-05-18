from abc import ABC
import numpy as np
import pandas as pd
import tensorflow as tf
from plugins.fastcompare.algo.algorithm_base import (
    AlgorithmBase,
    Parameter,
    ParameterType,
)


class HigherOrderEASE(AlgorithmBase, ABC):
    """Implementation of Higher-Order EASE algorithm for EasyStudy using ADMM optimization
    paper: https://dl.acm.org/doi/pdf/10.1145/3460231.3474273
    """

    def __init__(self, loader, positive_threshold, l2_B, l2_C, rho, m, **kwargs):
        self._ratings_df = loader.ratings_df
        self._loader = loader
        self._all_items = self._ratings_df.item.unique()

        self._rating_matrix = (
            self._loader.ratings_df.pivot(index="user", columns="item", values="rating")
            .fillna(0)
            .values
        )

        self._threshold = positive_threshold
        self._l2_B = l2_B
        self._l2_C = l2_C
        self._rho = rho
        self._m = m

        self._items_count = np.shape(self._rating_matrix)[1]

        self._pair_weights = None
        self._triplet_weights = None

    def _get_higher_order_data(self):
        # Apply threshold to item-item matrix to get most frequent pairs
        item_item_matrix = np.where(self._rating_matrix.T @ self._rating_matrix >= self._threshold, 1, 0)
        # Get indices of m most frequent pairs
        pair_indices = np.argpartition(item_item_matrix, -self._m, axis=None)[-self._m:]
        pair_indices = np.vstack(np.unravel_index(pair_indices, item_item_matrix.shape)).T

        # Create M matrix indicating selected pairs
        M = np.zeros((self._m, self._items_count))
        for idx, (i, j) in enumerate(pair_indices):
            M[idx, i] = 1
            M[idx, j] = 1

        # Generate higher-order training data Z
        Z = self._rating_matrix @ M.T
        Z = np.where(Z >= 2, 1, 0)

        return M, Z

    def fit(self):
        X = tf.convert_to_tensor(
            np.where(self._rating_matrix >= self._threshold, 1, 0), dtype=tf.float32
        )

        M, Z = self._get_higher_order_data()
        M = tf.convert_to_tensor(M, dtype=tf.float32)
        Z = tf.convert_to_tensor(Z, dtype=tf.float32)

        G = tf.transpose(X) @ X
        G += self._l2_B * tf.eye(self._items_count)

        P = tf.linalg.inv(G)

        # Initialize matrices
        B = tf.Variable(tf.zeros((self._items_count, self._items_count)))
        C = tf.Variable(tf.zeros((self._m, self._items_count)))
        D = tf.Variable(tf.zeros((self._m, self._items_count)))
        Gamma = tf.Variable(tf.zeros((self._m, self._items_count)))

        # ADMM iterations
        for _ in range(40):
            # Update B
            B_update = tf.subtract(tf.eye(self._items_count),
                                   P @ (tf.transpose(X) @ Z @ C - tf.linalg.diag(
                                       tf.reduce_sum(P @ (tf.transpose(X) @ Z @ C), axis=1))))
            B.assign(B_update)

            # Update C
            C_update = tf.linalg.inv(tf.transpose(Z) @ Z + (self._l2_C + self._rho) * tf.eye(self._m)) @ (
                    tf.transpose(Z) @ X @ (tf.eye(self._items_count) - B) + self._rho * (D - Gamma))
            C.assign(C_update)

            # Update D
            D_update = tf.multiply((1 - M), C)
            D.assign(D_update)

            # Update Gamma
            Gamma_update = Gamma + C - D
            Gamma.assign(Gamma_update)

        self._pair_weights = B
        self._triplet_weights = C

    def predict(self, selected_items, filter_out_items, k):
        rat = pd.DataFrame({"item": selected_items}).set_index("item", drop=False)
        candidates = np.setdiff1d(self._all_items, rat.item.unique())
        candidates = np.setdiff1d(candidates, filter_out_items)

        if not selected_items:
            return np.random.choice(candidates, size=k, replace=False).tolist()

        indices = list(selected_items)
        user_vector = np.zeros((self._items_count,))
        for i in indices:
            user_vector[i] = 1.0

        _, Z = self._get_higher_order_data()

        preds = (
                tf.tensordot(
                    tf.convert_to_tensor(user_vector, dtype=tf.float32), self._pair_weights, 1
                )
                + tf.tensordot(
            tf.convert_to_tensor(Z, dtype=tf.float32), self._triplet_weights, 1
        )
        ).numpy()

        candidate_scores = np.take(preds, candidates)
        top_indices = np.argsort(-candidate_scores)[:k]
        result = candidates[top_indices].tolist()

        return result

    @classmethod
    def name(cls):
        return "Higher-Order EASE"

    @classmethod
    def parameters(cls):
        return [
            Parameter(
                "l2_B",
                ParameterType.FLOAT,
                500.0,
                help="L2-norm regularization for pairwise weights",
            ),
            Parameter(
                "l2_C",
                ParameterType.FLOAT,
                500.0,
                help="L2-norm regularization for triplet weights",
            ),
            Parameter(
                "rho",
                ParameterType.FLOAT,
                1.0,
                help="Penalty parameter for ADMM",
            ),
            Parameter(
                "m",
                ParameterType.INT,
                40000,
                help="Number of triplet relations",
            ),
            Parameter(
                "positive_threshold",
                ParameterType.FLOAT,
                2.5,
                help="Threshold for conversion of n-ary rating into binary (positive/negative).",
            ),
        ]