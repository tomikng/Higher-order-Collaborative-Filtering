from abc import ABC
import numpy as np
import pandas as pd
import tensorflow as tf
from plugins.fastcompare.algo.algorithm_base import (
    AlgorithmBase,
    Parameter,
    ParameterType,
)


class HigherOrderModel(AlgorithmBase, ABC):
    def __init__(self, loader, l2_b, l2_c, rho, threshold, **kwargs):
        self._ratings_df = loader.ratings_df
        self._loader = loader
        self._all_items = self._ratings_df.item.unique()

        self._rating_matrix = (
            self._loader.ratings_df.pivot(index="user", columns="item", values="rating")
            .fillna(0)
            .values
        )

        self._l2_b = l2_b
        self._l2_c = l2_c
        self._rho = rho
        self._threshold = threshold

        self._num_users, self._num_items = np.shape(self._rating_matrix)
        self._pairwise_matrix = None
        self._higher_order_matrix = None

    def fit(self):
        X = tf.convert_to_tensor(
            np.where(self._rating_matrix >= self._threshold, 1, 0), dtype=tf.float32
        )
        X_T = tf.transpose(X)

        # Compute pairwise relations
        G = X_T @ X + self._l2_b * tf.eye(self._num_items)
        P = tf.linalg.inv(G)
        B = P / (-tf.linalg.tensor_diag_part(P))
        B = tf.linalg.set_diag(B, tf.zeros(B.shape[0]))

        # Compute higher-order relations
        Z, actual_num_relations = self._create_higher_order_matrix(X)
        C = self._train_higher_order(X, Z, B, actual_num_relations)

        self._pairwise_matrix = B.numpy()
        self._higher_order_matrix = C.numpy()

    def _create_higher_order_matrix(self, X):
        max_relations = 40000  # Example, adjust this according to needs
        Z = np.zeros((self._num_users, max_relations))

        relation_counter = 0
        for user_index in range(self._num_users):
            interactions = np.nonzero(X[user_index, :])[0]
            if len(interactions) < 2:
                continue
            for i in range(len(interactions)):
                for j in range(i + 1, len(interactions)):
                    if relation_counter >= max_relations:
                        break
                    # Simplified higher-order interaction (e.g., pair of items)
                    Z[user_index, relation_counter] = 1
                    relation_counter += 1

        return tf.convert_to_tensor(Z[:, :relation_counter], dtype=tf.float32), relation_counter

    def _train_higher_order(self, X, Z, B, num_relations):
        m = num_relations
        C = tf.Variable(tf.zeros((m, self._num_items)), dtype=tf.float32)
        D = tf.Variable(tf.zeros((m, self._num_items)), dtype=tf.float32)
        Γ = tf.Variable(tf.zeros((m, self._num_items)), dtype=tf.float32)

        for _ in range(40):  # Number of ADMM iterations
            B_inv = tf.linalg.inv(tf.transpose(X) @ X + self._l2_b * tf.eye(self._num_items))
            C.assign(B_inv @ (tf.transpose(X) @ (X - Z @ C) - Γ / self._rho))

            D.assign(tf.where(Z == 0, C, 0))

            Γ.assign(Γ + self._rho * (C - D))

        return C

    def predict(self, selected_items, filter_out_items, k):
        rat = pd.DataFrame({"item": selected_items}).set_index("item", drop=False)
        candidates = np.setdiff1d(self._all_items, rat.item.unique())
        candidates = np.setdiff1d(candidates, filter_out_items)

        if not selected_items:
            return np.random.choice(candidates, size=k, replace=False).tolist()

        user_vector = np.zeros((self._num_items,))
        for item in selected_items:
            user_vector[item] = 1.0

        user_vector_tf = tf.convert_to_tensor(user_vector, dtype=tf.float32)
        pairwise_scores = tf.tensordot(user_vector_tf, self._pairwise_matrix, 1).numpy()
        higher_order_scores = tf.tensordot(user_vector_tf, self._higher_order_matrix, 1).numpy()

        combined_scores = pairwise_scores + higher_order_scores
        candidates_by_prob = sorted(((combined_scores[cand], cand) for cand in candidates), reverse=True)
        result = [x for _, x in candidates_by_prob][:k]

        return result

    @classmethod
    def name(cls):
        return "HigherOrderModel"

    @classmethod
    def parameters(cls):
        return [
            Parameter("l2_b", ParameterType.FLOAT, 0.1, help="L2-norm regularization for B"),
            Parameter("l2_c", ParameterType.FLOAT, 0.1, help="L2-norm regularization for C"),
            Parameter("rho", ParameterType.FLOAT, 1.0, help="Penalty parameter for ADMM"),
            Parameter("threshold", ParameterType.FLOAT, 0.5, help="Threshold for binary rating conversion"),
        ]
