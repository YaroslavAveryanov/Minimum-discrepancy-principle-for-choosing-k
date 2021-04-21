# Minimum discrepancy principle strategy for choosing k in kNN regression
# A class that describes the kNN regression estimator

import sys
import scipy as sp
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import time
from complementary import *




class kNN_estimator:

    '''
    A class that performs model model selection for choosing k in the kNN regression estimator.

    The non-parametric regression model:
        y_i = f^*(x_i) + \varepsilon_i, i = 1, \ldots, n;

    Parameters
    ----------

    fl_data : bool, default=False
        0 if artificial data, and 1 if real data used

    k_max : int, default=10
        The maximum number of the nearest neighbors for the kNN regression estimator

    f_star : array-like of shape (n_samples,), default=None
        The vector of the regression function on the training data; used only with artificial data

    sigma : float, default=None
        The value (or an estimator) of noise (standard deviation) in the regression model

    '''

    def __init__(self, fl_data=0, k_max=10, f_star=None, sigma=None):

        self.fl_data = fl_data
        self.k_max = k_max
        self.f_star = f_star
        self.sigma = sigma



    def oracle_and_star(self, x, y):
        '''
        Returns the oracle and bias-variance trade-off selection rules, and
        the empirical norm for each k \in {1, ..., k_max}
        Parameters
        ----------

        x : array-like of shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features

        y : array-like of shape (n_samples,)
            The target variable for the regression problem

        '''

        #Vector of biases, variances, and expectations of empirical risk
        bias = np.zeros(self.k_max)
        variance = np.zeros(self.k_max)
        exp_empirical_risk = np.zeros(self.k_max)

        #Vector of the empirical norm evaluated at each k
        risk = np.zeros(self.k_max)

        #Real data
        if (self.fl_data):
            k_or = None
            k_star = None
        #Artificial data
        else:
            k_or = 0
            k_star = 0

            increm = np.linspace(0, len(x)-1, len(x)).astype(int)

            k = self.k_max
            [A, ind] = fill_matrix_2d(x, k)


            variance[0] = (self.sigma)**2

            while(k != 1):

                L = np.identity(len(x)) - A

                bias[k-1] = np.linalg.norm(np.dot(L, self.f_star))**2 / len(x)
                variance[k-1] = (self.sigma)**2 / k


                risk[k-1] = np.linalg.norm(np.dot(A, y) - self.f_star)**2 / len(x)


                exp_empirical_risk[k-1] = np.linalg.norm(np.dot(L, self.f_star))**2 / len(x) + self.sigma**2 / len(x)  * np.trace(np.dot(np.matrix.transpose(L), L))


                if (exp_empirical_risk[k-1] >= (self.sigma)**2):
                    k_star = k


                if (k != 1):
                    A[increm, ind[:, k-1]] = 0
                    A = k / (k - 1) * A

                k = k - 1


            if (k_star == 0):
                k_star = self.k_max

            k_or = oracle(bias + variance) - 1
            k_star = k_star - 1

            return [k_or, k_star, risk]



    def minimum_discrepancy(self, x, y):
        '''
        Minimum discrepancy principle strategy for model selection with the kNN regression estimator;
        Returns the chosen k and runtime of the procedure

        Parameters
        ----------

        x : array-like of shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features

        y : array-like of shape (n_samples,)
            The target variable for the regression problem

        '''

        start_mdp_ = time.time()

        n = len(y)
        increm = np.linspace(0, n-1, n).astype(int)

        empirical_risk = np.zeros(self.k_max)
        k = self.k_max
        k_opt = self.k_max
        fl = 1

        while(k != 1 and fl):

            #Computes first the estimator of \sigma
            if (k == self.k_max):
                [A, distance_indices] = fill_matrix_2d(x, k)

                A_2 = np.zeros((n,n))

                A_2[increm, distance_indices[:, 1]] = 1/2
                A_2[increm, distance_indices[:, 0]] = 1/2

                sigma_est = np.sqrt(sigma_sq_est(y, A_2, 2))


            empirical_risk[k-1] = np.linalg.norm(y - np.dot(A, y))**2 / n

            #Stop the learing process if the empirical risk crosses \sigma**2
            if (empirical_risk[k-1] <= sigma_est**2):
                k_opt = k
                fl = 0

            if (k != 1):
                A[increm, distance_indices[:, k-1]] = 0
                A = k / (k - 1) * A


            k = k - 1

        if (k == 1):
            k_opt = 2


        end_mdp_ = time.time()


        return [k_opt - 1, end_mdp_ - start_mdp_]


    def gcv(self, x, y):
        '''
        Generalized cross-validation for the kNN regression estimator;
        Returns the chosen k and runtime of the procedure

        Parameters
        ----------

        x : array-like of shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features

        y : array-like of shape (n_samples,)
            The target variable for the regression problem
        '''

        start_gcv = time.time()

        n = len(x)
        increm = np.linspace(0, n-1, n).astype(int)

        pred_k = np.zeros(self.k_max)
        k = self.k_max


        [A, distance_indices] = fill_matrix_2d(x, self.k_max)

        while(k != 1):

            pred_k[k-1] = (np.linalg.norm(y - np.dot(A, y))**2 / n) / (1 - np.trace(A) / n)**2

            A[increm, distance_indices[:, k-1]] = 0

            A = k / (k - 1) * A
            k = k - 1


        k_gcv = np.argmin(pred_k[1:]) + 1

        end_gcv = time.time()

        return [k_gcv, end_gcv - start_gcv]


    def holdout(self, x, y):
        '''
        The hold-out selection rule for the kNN regression estimator;
        Returns the chosen k and runtime of the procedure

        Parameters
        ----------

        x : array-like of shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features

        y : array-like of shape (n_samples,)
            The target variable for the regression problem

        '''

        start_ho = time.time()

        x_tr, x_te, y_tr, y_te = train_test_split(x, y, test_size=0.5, random_state=42)

        n_tr = x_tr.shape[0]
        n_te = x_te.shape[0]

        pred_k = np.zeros(self.k_max)

        k = self.k_max
        increm = np.linspace(0, n_te-1, n_te).astype(int)


        [distances, distance_indices] = compute_distances_cross(x_te, x_tr)

        while(k != 1):
            y_pred = np.zeros(n_te)

            if (k == self.k_max):
                [A, distance_indices] = fill_matrix_cross(x_te, x_tr, k)


            pred_k[k-1] = np.linalg.norm(y_te - np.dot(A, y_tr))**2

            if (k != 1):
                A[increm, distance_indices[:, k-1]] = 0
                A = k / (k - 1) * A


            k = k - 1

        k_ho = np.argmin(pred_k[1:]) + 1

        end_ho = time.time()

        return [k_ho, end_ho - start_ho]


    def aic(self, x, y):
        '''
        Akaike's AIC model selection rule for the kNN regression estimator;
        Returns the chosen k and runtime of the procedure

        Parameters
        ----------

        x : array-like of shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features

        y : array-like of shape (n_samples,)
            The target variable for the regression problem

        '''

        start_aic_ = time.time()

        n = len(x)
        increm = np.linspace(0, n-1, n).astype(int)

        k = self.k_max
        pred_k = np.zeros(self.k_max)

        while(k != 1):

            if (k == self.k_max):
                [A, distance_indices] = fill_matrix_2d(x, k)

                A_2 = np.zeros((n , n))

                A_2[increm, distance_indices[:, 1]] = 1/2
                A_2[increm, distance_indices[:, 0]] = 1/2


                sigma_est = np.sqrt(sigma_sq_est(y, A_2, 2))

            pred_k[k-1] = 1 / (n * sigma_est**2) * (np.linalg.norm(y - np.dot(A, y))**2 + 2 * np.trace(A) * sigma_est**2)

            A[increm, distance_indices[:, k-1]] = 0

            A = k / (k - 1) * A

            k = k - 1


        k_aic = np.argmin(pred_k[1:]) + 1

        end_aic_ = time.time()

        return [k_aic, end_aic_ - start_aic_]



    def five_fold_cv(self, x, y):
        '''
        The 5-fold cross-validation selection rule for the kNN regression estimator

        Parameters
        ----------

        x : array-like of shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features

        y : array-like of shape (n_samples,)
            The target variable for the regression problem

        '''

        start_cv_ = time.time()

        kf = KFold(n_splits=5)

        error_cur = 0
        pred_k = np.zeros(self.k_max)


        i = 0
        for train_index, test_index in kf.split(x):

            k = self.k_max

            X_train, X_test = x[train_index], x[test_index]
            y_train, y_test = y[train_index], y[test_index]
            n_tr = len(X_train)
            n_te = len(X_test)

            increm = np.linspace(0, n_te-1, n_te).astype(int)

            [distances, distance_indices] = compute_distances_cross(X_test, X_train)

            while(k != 1):

                if (k == self.k_max):
                    [A, distance_indices] = fill_matrix_cross(X_test, X_train, k)


                pred_k[k-1] += np.linalg.norm(y_test - np.dot(A, y_train))**2 / 5

                if (k != 1):
                    A[increm, distance_indices[:, k-1]] = 0

                    A = k / (k - 1) * A
                    k = k - 1

            i = i + 1


        k_5f = np.argmin(pred_k[1:]) + 1

        end_cv_ = time.time()


        return [k_5f, end_cv_ - start_cv_]


    def predict(self, x, y, x_test, k_hat):
        '''
        Making prediction on the test test x_test for the kNN regression estimator trained on (x, y) and with k=k_hat;
        Returns the vector of predictions

        Parameters
        ----------

        x : array-like of shape (n_samples_1, n_features)
            Training data, where n_samples_1 is the number of samples
            and n_features is the number of features

        y : array-like of shape (n_samples_1,)
            The target variable for the regression problem

        x_test : array-like of shape (n_samples_2, n_features)
                 Test data, where n_samples_2 is the number of samples
                 and n_features is the number of features

        k_hat : int,
            The number of nearest neighbors chosen by a model selection procedure


        '''

        n_te = len(x_test)
        y_pred = np.zeros(n_te)

        [distances, distance_indices] = compute_distances_cross(x_test, x)

        for j in range(n_te):

            w_at_j = np.zeros(len(y))

            distance_indices_knn = distance_indices[j,:k_hat]

            w_at_j[distance_indices_knn] = 1/(k_hat)

            y_pred[j] = np.dot(w_at_j, y)

        return y_pred


#---------------------------------------------------------------------------------------------------------------
