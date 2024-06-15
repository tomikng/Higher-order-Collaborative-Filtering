import numpy as np
from scipy import sparse
import pandas as pd
from copy import deepcopy
from abc import ABC
from server.plugins.fastcompare.algo.algorithm_base import (
    AlgorithmBase,
    Parameter,
    ParameterType,
)


class HigherOrderCollaborativeFiltering(AlgorithmBase, ABC):
    def __init__(self, loader, threshold, lambdaBB, lambdaCC, rho, epochs, **kwargs):
        print("Initializing HigherOrderCollaborativeFiltering")
        self._ratings_df = loader.ratings_df
        self._loader = loader
        self._all_items = self._ratings_df['item'].unique()

        # Select a subset of users for training
        # self._ratings_df, self._unique_uid, self._unique_sid = self.prepare_training_data(self._ratings_df)

        self._rating_matrix = self.create_rating_matrix(self._ratings_df)
        self._threshold = threshold
        self._lambdaBB = lambdaBB
        self._lambdaCC = lambdaCC
        self._rho = rho
        self._epochs = epochs

        self._items_count = np.shape(self._rating_matrix)[1]

        self._XtX = None
        self._XtXdiag = None
        self._BB = None
        self._CC = None

    def create_rating_matrix(self, ratings_df):
        print("Creating rating matrix")
        rating_matrix = (
            ratings_df.pivot(index="user", columns="item", values="rating")
            .fillna(0)
            .values
        )
        return rating_matrix

    def fit(self):
        print("Starting fit process")
        X = self._rating_matrix
        self._XtX = X.T @ X
        self._XtXdiag = deepcopy(np.diag(self._XtX))
        self._XtX[np.diag_indices(self._XtX.shape[0])] = self._XtXdiag

        print("Creating list of feature pairs")
        ii_feature_pairs = self.create_list_feature_pairs(self._XtX, self._threshold)
        print(f"Number of feature pairs: {len(ii_feature_pairs[0])}")

        print("Creating matrix Z and CCmask")
        Z, CCmask = self.create_matrix_Z(ii_feature_pairs, X)
        print(f"Matrix Z shape: {Z.shape}, CCmask shape: {CCmask.shape}")

        if Z.shape[0] == 0:  # Handle empty Z case
            self._BB = np.zeros((self._XtX.shape[0], self._XtX.shape[1]), dtype=np.float64)
            self._CC = np.zeros((0, self._XtX.shape[0]), dtype=np.float64)
            print("No feature pairs found, returning zero matrices for BB and CC")
            return

        print("Creating higher-order matrices")
        ZtZ = Z.T @ Z
        ZtX = Z.T @ X
        ZtZdiag = deepcopy(np.diag(ZtZ))

        print("Training higher-order model")
        self._BB, self._CC = self.train_higher(self._XtX, self._XtXdiag, self._lambdaBB, ZtZ, ZtZdiag, self._lambdaCC,
                                               CCmask, ZtX, self._rho, self._epochs)
        print(f"Fit process completed with BB shape: {self._BB.shape} and CC shape: {self._CC.shape}")

    def predict(self, selected_items, filter_out_items, k):
        print("Starting prediction")
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

        candidates_by_prob = sorted(((preds[cand], cand) for cand in candidates), reverse=True)
        result = [x for _, x in candidates_by_prob][:k]

        return result

    @classmethod
    def name(cls):
        return "Higher Order CF"

    @classmethod
    def parameters(cls):
        return [
            Parameter("threshold", ParameterType.FLOAT, 100000, help="Threshold for feature pair creation"),  # Adjusted to have cca 500 pairs
            Parameter("lambdaBB", ParameterType.FLOAT, 500, help="Lambda parameter for BB regularization"),
            Parameter("lambdaCC", ParameterType.FLOAT, 2000, help="Lambda parameter for CC regularization"),
            Parameter("rho", ParameterType.FLOAT, 30000, help="Rho parameter for CC regularization"),
            Parameter("epochs", ParameterType.INT, 40, help="Number of epochs for training")
        ]

    @staticmethod
    def create_list_feature_pairs(XtX, threshold):
        print("Creating list of feature pairs")
        AA = np.triu(np.abs(XtX))
        AA[np.diag_indices(AA.shape[0])] = 0.0
        ii_pairs = np.where((AA > threshold) == True)
        print(f"Number of feature pairs created: {len(ii_pairs[0])}")
        return ii_pairs

    @staticmethod
    def create_matrix_Z(ii_pairs, X):
        print("Starting create_matrix_Z")

        if len(ii_pairs[0]) == 0:
            print("No feature pairs found, Z will be empty")
            return np.zeros((0, X.shape[1]), dtype=np.float64), np.zeros((0, X.shape[1]), dtype=np.float64)

        MM = np.zeros((len(ii_pairs[0]), X.shape[1]), dtype=np.float64)
        MM[np.arange(MM.shape[0]), ii_pairs[0]] = 1.0
        MM[np.arange(MM.shape[0]), ii_pairs[1]] = 1.0

        CCmask = 1.0 - MM
        MM = sparse.csc_matrix(MM.T)

        Z = X @ MM
        Z = (Z == 2.0)
        Z = Z * 1.0
        print(f"Matrix Z created with shape: {Z.shape}")
        print(f"CCmask created with shape: {CCmask.shape}")

        return Z, CCmask

    @staticmethod
    def train_higher(XtX, XtXdiag, lambdaBB, ZtZ, ZtZdiag, lambdaCC, CCmask, ZtX, rho, epochs):
        print("Starting training of higher-order model")
        ii_diag = np.diag_indices(XtX.shape[0])
        XtX[ii_diag] = XtXdiag + lambdaBB
        PP = np.linalg.inv(XtX)

        ii_diag_ZZ = np.diag_indices(ZtZ.shape[0])
        ZtZ[ii_diag_ZZ] = ZtZdiag + lambdaCC + rho
        QQ = np.linalg.inv(ZtZ)

        BB = np.zeros((XtX.shape[0], XtX.shape[1]), dtype=np.float64)
        if ZtZ.shape[0] == 0:
            print("ZtZ is empty, returning zero matrices")
            return BB, np.zeros((0, XtX.shape[0]), dtype=np.float64)

        CC = np.zeros((ZtZ.shape[0], XtX.shape[0]), dtype=np.float64)
        DD = np.zeros((ZtZ.shape[0], XtX.shape[0]), dtype=np.float64)
        UU = np.zeros((ZtZ.shape[0], XtX.shape[0]), dtype=np.float64)

        for iter in range(epochs):
            print(f"Epoch {iter}")
            XtX[ii_diag] = XtXdiag
            BB = PP.dot(XtX - ZtX.T @ CC)
            gamma = np.diag(BB) / np.diag(PP)
            BB -= PP * gamma[:, None]

            CC = QQ.dot(ZtX - ZtX @ BB + rho * (DD - UU))
            DD = CC * CCmask
            UU += CC - DD

            print(f"End of epoch {iter} - CC shape: {CC.shape}, DD shape: {DD.shape}, UU shape: {UU.shape}")

        print("Training completed")
        return BB, DD
