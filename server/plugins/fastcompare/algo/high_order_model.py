from abc import ABC
import numpy as np
import tensorflow as tf

from server.plugins.fastcompare.algo.algorithm_base import ParameterType, Parameter
from server.plugins.fastcompare.algo.ease import EASE


class HigherOrderEASE(EASE, ABC):
    """Implementation of Higher-Order EASE algorithm for EasyStudy using ADMM optimization
    paper: https://dl.acm.org/doi/pdf/10.1145/3460231.3474273
    """

    def __init__(self, loader, positive_threshold, l2, l2_C, rho, m, **kwargs):
        super().__init__(loader, positive_threshold, l2, **kwargs)
        self._M = None
        self._C = None
        self._l2_C = l2_C
        self._rho = rho
        self._m = m

    def fit(self):
        print("Training Higher-Order EASE model...")
        super().fit()

        # Reference: Section 3.1 Data Representation
        X = tf.convert_to_tensor(
            np.where(self._rating_matrix >= self._threshold, 1, 0), dtype=tf.float32
        )
        item_item_matrix = tf.transpose(X) @ X
        S = self._select_higher_order_relations(item_item_matrix, self._m)
        self._M = self._create_M_matrix(S)

        # Reference: Section 3.2 Training Objective and Section 3.3 Update Equations for Training
        Z = self._generate_higher_order_data(X, self._M)
        self._C = self._train_higher_order_model(X, Z, self._M)
        print("Higher-Order EASE model trained successfully.")

    def _select_higher_order_relations(self, item_item_matrix, m):
        """Selects the top m higher-order relations based on the item-item co-occurrence matrix."""

        item_item_matrix = item_item_matrix.numpy()

        upper_tri_indices = np.triu_indices(self._items_count, k=1)

        upper_tri_values = item_item_matrix[upper_tri_indices]

        threshold_value = np.partition(upper_tri_values, -m)[-m]

        # Select the item pairs that meet or exceed the threshold
        selected_pairs_indices = np.where(upper_tri_values >= threshold_value)

        # Gather the corresponding upper triangular indices for the selected pairs
        selected_pairs = set(
            (int(upper_tri_indices[0][idx]), int(upper_tri_indices[1][idx]))
            for idx in selected_pairs_indices[0]
        )

        print(f"Selected {len(selected_pairs)} higher-order relations.")

        return selected_pairs

    def _create_M_matrix(self, S):
        """
        Creates the M matrix from the set of selected higher-order relations S.
        """
        print("Creating M matrix...")
        M = np.zeros((len(S), self._items_count), dtype=np.float32)
        for r, (i, k) in enumerate(S):
            M[r, i] = 1
            M[r, k] = 1
        return M

    def _generate_higher_order_data(self, X, M):
        # Reference: Section 3.1 Data Representation
        print("Generating higher-order data...")
        M = tf.convert_to_tensor(M, dtype=tf.float32)  # Convert M to float32
        Z = tf.matmul(X, tf.transpose(M))
        Z = tf.where(Z >= 2.5, 1, 0)  # Apply thresholding
        return Z

    def _train_higher_order_model(self, X, Z, M):
        # Reference: Section 3.3 Update Equations for Training
        print("Training higher-order model using ADMM")
        B = self._weights
        C = tf.zeros((M.shape[0], self._items_count), dtype=tf.float32)
        D = tf.zeros((M.shape[0], self._items_count), dtype=tf.float32)
        Gamma = tf.zeros((M.shape[0], self._items_count), dtype=tf.float32)

        for i in range(40):  # Run ADMM for 40 iterations
            print(f"Iteration {i}...")
            B = self._update_B(X, Z, C)
            C = self._update_C(X, Z, B, D, Gamma)
            D = self._update_D(C, M)
            Gamma = self._update_Gamma(C, D, Gamma)

        self._weights = B
        return C

    def _update_B(self, X, Z, C):
        # Reference: Equation 10-12
        print("Updating B matrix...")
        P = tf.linalg.inv(tf.transpose(X) @ X + self._l2 * tf.eye(self._items_count, dtype=tf.float32))
        B = tf.eye(self._items_count, dtype=tf.float32) - P @ (
                tf.transpose(X) @ tf.cast(Z, dtype=tf.float32) @ tf.cast(C, dtype=tf.float32)
                - tf.linalg.diag(tf.reduce_sum(
            P @ tf.transpose(X) @ tf.cast(Z, dtype=tf.float32) @ tf.cast(C, dtype=tf.float32),
            axis=0))
        )
        B = tf.linalg.set_diag(B, tf.zeros(self._items_count, dtype=tf.float32))
        return B

    def _update_C(self, X, Z, B, D, Gamma):
        # Reference: Equation 13
        print("Updating C matrix...")

        # Ensure Z is cast to float32
        Z = tf.cast(Z, dtype=tf.float32)

        # Compute the left term: (Z^T @ Z + (λ_C + ρ) * I)^-1
        left_term = tf.linalg.inv(
            tf.transpose(Z) @ Z + (self._l2_C + self._rho) * tf.eye(Z.shape[1], dtype=tf.float32)
        )

        # Compute the right term: Z^T @ X @ (I - B) + ρ * (D - Γ)
        right_term = tf.transpose(Z) @ X @ (tf.eye(self._items_count, dtype=tf.float32) - B) + self._rho * (D - Gamma)

        # Compute the updated C matrix
        C_updated = left_term @ right_term

        return C_updated

    def _update_D(self, C, M):
        # Reference: Equation 8
        print("Updating D matrix...")
        return (1 - M) * C

    def _update_Gamma(self, C, D, Gamma):
        # Reference: Equation 9
        print("Updating Gamma matrix...")
        return Gamma + C - D

    def predict(self, selected_items, filter_out_items, k):
        print("Generating predictions using Higher-Order EASE model...")

        # Section 3.5
        user_vector = np.zeros((self._items_count,), dtype=np.float32)
        for i in selected_items:
            user_vector[i] = 1.0

        # Ensure self._M is of type float32
        M = tf.convert_to_tensor(self._M, dtype=tf.float32)

        # Reshape user_vector to be 2D (self._items_count, 1) for matrix multiplication
        user_vector_2d = tf.expand_dims(user_vector, axis=-1)

        # Print shapes for debugging
        print(f"user_vector shape: {user_vector.shape}")
        print(f"user_vector_2d shape: {user_vector_2d.shape}")
        print(f"M shape: {M.shape}")

        # Perform matrix multiplication
        Z_u = tf.matmul(M, user_vector_2d)

        # Squeeze Z_u to remove the last dimension
        Z_u = tf.squeeze(Z_u, axis=-1)

        # Print shape of Z_u for debugging
        print(f"Z_u shape: {Z_u.shape}")

        # Calculate predictions
        preds = tf.tensordot(
            tf.convert_to_tensor(user_vector, dtype=tf.float32), self._weights, 1
        ) + tf.tensordot(
            Z_u, self._C, 1
        )

        # Print shapes for debugging
        print(f"self._weights shape: {self._weights.shape}")
        print(f"self._C shape: {self._C.shape}")
        print(f"preds shape: {preds.shape}")

        candidates = np.setdiff1d(self._all_items, selected_items)
        candidates = np.setdiff1d(candidates, filter_out_items)

        preds_candidates = preds.numpy()[candidates]
        top_k_indices = np.argsort(preds_candidates)[::-1][:k]
        result = candidates[top_k_indices].tolist()

        print(f"Generated {k} predictions using Higher-Order EASE model.")
        return result

    @classmethod
    def name(cls):
        return "Higher-Order EASE"

    @classmethod
    def parameters(cls):
        base_params = super().parameters()
        higher_order_params = [
            Parameter(
                "l2_C",
                ParameterType.FLOAT,
                0.1,
                help="L2-norm regularization for higher-order weight matrix C",
            ),
            Parameter(
                "rho",
                ParameterType.FLOAT,
                10.0,
                help="Penalty parameter for ADMM optimization",
            ),
            Parameter(
                "m",
                ParameterType.INT,
                500,
                help="Number of higher-order relations to consider",
            ),
        ]
        return base_params + higher_order_params
