# Minimum discrepancy principle strategy for choosing k in kNN regression


import sys
import scipy as sp
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib as mpl
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import load_boston, load_diabetes
from sklearn.datasets import fetch_california_housing
from knn_estimator_class import *



sns.set()



def main():

    fl_real_data = input("Type 0 if artificial data, 1 if real data: \n")

    fl_real_data = int(fl_real_data)
    if (fl_real_data != 0 and fl_real_data != 1):
        raise ValueError('Not an appropriate number')

    #Artificial data
    if (fl_real_data == 0):

        fl_func = input("Type 0 if smooth function, 1 if sinus function: \n")

        fl_func = int(fl_func)


        if (fl_func != 0 and fl_func != 1):
            raise ValueError('Not an appropriate function or data')

        sigma = input("Type the value of noise (std): \n")

        sigma = float(sigma)

        if (sigma < 0):
            raise ValueError('Not an appropriate value of noise')

        N_samples = input("Type the number of repetitions to perform: \n")

        N_samples = int(N_samples)

        if (N_samples <= 0):
            raise ValueError('Not an appropriate value of repetitions')


        #The vector of sample size
        n_array = np.array([50, 80, 100, 160, 200, 250])


        risks_or = np.zeros(len(n_array))
        risks_tau = np.zeros(len(n_array))
        risks_holdout = np.zeros(len(n_array))
        risks_star = np.zeros(len(n_array))
        risks_gcv = np.zeros(len(n_array))


        K_max = np.zeros(len(n_array)).astype(int)

        for v in range(len(n_array)):

            #The vector of k_max = sqrt(n)
            K_max[v] = int(np.sqrt(n_array[v]))

            #Data and target
            X = np.random.uniform(0, 1, 3 * n_array[v]).reshape((n_array[v], 3))
            generated_data = np.zeros((N_samples, n_array[v]))

            #Vector corresponding to the regression function
            f_star = np.zeros(n_array[v])


            for i in range(n_array[v]):
                #Smooth regression function
                if (fl_func == 0):
                    f_star[i] = 1.5 * (np.linalg.norm(X[i] - 0.5) / np.sqrt(X.shape[1]) - 0.5)
                #Sinus regression function
                elif (fl_func == 1):
                    f_star[i] = 1.5 * np.sin(np.linalg.norm(X[i]) / np.sqrt(X.shape[1]))


            for q in range(N_samples):
                epsilon = np.random.normal(0, 1, n_array[v])
                generated_data[q] = f_star + sigma * epsilon


                kNN_estimator_ = kNN_estimator(fl_real_data, K_max[v], f_star, sigma)

                #Model selection rules: oracle, star, MDP, GCV, and holdout
                [k_or_, k_star_, risk] = kNN_estimator_.oracle_and_star(X, generated_data[q])
                k_mdp_ = kNN_estimator_.minimum_discrepancy(X, generated_data[q])[0]
                k_gcv_ = kNN_estimator_.gcv(X, generated_data[q])[0]
                k_ho_ = kNN_estimator_.holdout(X, generated_data[q])[0]

                #Computes the average prediction error
                risks_or[v] += risk[int(k_or_)] / N_samples
                risks_star[v] += risk[int(k_star_) ] / N_samples
                risks_tau[v] += risk[int(k_mdp_)] / N_samples
                risks_holdout[v] += risk[int(k_ho_)] / N_samples
                risks_gcv[v] += risk[int(k_gcv_)] / N_samples



        print("pred mdp =", risks_tau)
        print("pred gcv =", risks_gcv)
        print("pred ho =", risks_holdout)
        print("pred star =", risks_star)
        print("pred or =", risks_or)

        #Plot all results for the prediction error
        mpl.use('tkagg')
        plt.figure(figsize=(11, 6))
        plt.style.use('ggplot')
        plt.plot(n_array, risks_or, 'r', label = r'$k_{or}$', linewidth=2)
        plt.plot(n_array, risks_tau, 'b:', label = r'$k^{\tau}$', linewidth=2)
        plt.plot(n_array, risks_star, 'm.-', label = r'$k^*$', linewidth=2)
        plt.plot(n_array, risks_holdout, 'y--', label = r'$k_{HO}$', linewidth=2)
        plt.plot(n_array, risks_gcv, 'g-.', label = r'$k_{GCV}$', linewidth=2)

        if (fl_func == 0):
            func_caption = ', smooth.r.f.'
        elif (fl_func == 1):
            func_caption = ', sinus.r.f.'


        plt.legend(loc = 1, fontsize=20)
        plt.title('KNN regressor,' + r' $\sigma$ ' + '= %.2f'%sigma +  func_caption, fontsize=19)
        plt.xlabel("Sample size", fontsize=17)
        plt.ylabel("Average loss", fontsize=17)
        plt.tight_layout()
        plt.show()

    #Real data
    elif (fl_real_data == 1):

        fl_data = input("Type the data you want to choose:\nType 0 if Boston House Prices data,\n1 if Diabetes data,\n2 if California House Prices data,\n3 if Power Plants data:\n")
        fl_data = int(fl_data)

        if (fl_data != 0 and fl_data != 1 and fl_data != 2 and fl_data != 3):
            raise ValueError("Not an appropriate value for data")

        #Boston Housing Prices data
        if (fl_data == 0):
            dataset = load_boston()
            X = dataset.data
            Y = dataset.target
        #Diabetes data
        if (fl_data == 1):
            dataset = load_diabetes()
            X = dataset.data
            Y = dataset.target
        #California Housing Prices data
        if (fl_data == 2):
            dataset = fetch_california_housing()
            X = dataset.data[:3000]
            Y = dataset.target[:3000]
        #Power Plants data
        if (fl_data == 3):
            dataset = pd.read_excel("Power Plant/Folds5x2_pp.xlsx")
            Y = np.array(dataset['PE'])[:3000]
            X = dataset.drop(columns=['PE'])[:3000]


        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

        #Make scaling
        scaler = MinMaxScaler()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)

        X_test = scaler.transform(X_test)

        n = len(X_train)

        grid_of_samples = [n//5, n//4, n//3, n//2, n]
        grid_of_k = [3 * int(np.log(n/5)), 3 * int(np.log(n/4)), 3 * int(np.log(n/3)), 3 * int(np.log(n/2)), 3 * int(np.log(n))]



        pred_k_mdp = np.zeros(len(grid_of_samples))
        pred_k_gcv = np.zeros(len(grid_of_samples))
        pred_k_aic = np.zeros(len(grid_of_samples))
        pred_k_ho = np.zeros(len(grid_of_samples))
        pred_k_5f = np.zeros(len(grid_of_samples))


        time_mdp = np.zeros(len(grid_of_samples))
        time_gcv = np.zeros(len(grid_of_samples))
        time_ho = np.zeros(len(grid_of_samples))
        time_5f = np.zeros(len(grid_of_samples))
        time_aic = np.zeros(len(grid_of_samples))

        for v in range(len(grid_of_samples)):

            for q in range(25):

                rand_indices_ = np.random.choice(n, grid_of_samples[v], replace=False)

                X_train_q = X_train[rand_indices_]
                y_train_q = y_train[rand_indices_]


                kNN_estimator_ = kNN_estimator(fl_data=fl_real_data, k_max=grid_of_k[v])

                #Results for the minimum discrepancy principle
                mdp_results_ = kNN_estimator_.minimum_discrepancy(X_train_q, y_train_q)

                k_mdp_ = mdp_results_[0]
                time_mdp[v] += mdp_results_[1]



                y_pred_mdp = kNN_estimator_.predict(X_train_q, y_train_q, X_test, k_mdp_)
                pred_k_mdp[v] += np.linalg.norm(y_pred_mdp - y_test)

                #Results for the generalized cross-validation
                gcv_results_ = kNN_estimator_.gcv(X_train_q, y_train_q)

                k_gcv_ = gcv_results_[0]
                time_gcv[v] += gcv_results_[1]


                y_pred_gcv = kNN_estimator_.predict(X_train_q, y_train_q, X_test, k_gcv_)
                pred_k_gcv[v] += np.linalg.norm(y_pred_gcv - y_test)

                #Results for the AIC selection rule
                aic_results_ = kNN_estimator_.aic(X_train_q, y_train_q)

                k_aic_ = aic_results_[0]
                time_aic[v] += aic_results_[1]


                y_pred_aic = kNN_estimator_.predict(X_train_q, y_train_q, X_test, k_aic_)
                pred_k_aic[v] += np.linalg.norm(y_pred_aic - y_test)

                #Results for the 5-fold cross-validation selection rule
                five_fold_results_ = kNN_estimator_.five_fold_cv(X_train_q, y_train_q)

                k_5f_ = five_fold_results_[0]
                time_5f[v] += five_fold_results_[1]


                y_pred_5f = kNN_estimator_.predict(X_train_q, y_train_q, X_test, k_5f_)
                pred_k_5f[v] += np.linalg.norm(y_pred_5f - y_test)

            #Make an average of all predictions and runtime
            pred_k_mdp[v] = pred_k_mdp[v] / 25
            pred_k_gcv[v] = pred_k_gcv[v] / 25
            pred_k_aic[v] = pred_k_aic[v] / 25
            pred_k_5f[v] = pred_k_5f[v] / 25

            time_mdp[v] = time_mdp[v] / 25
            time_gcv[v] = time_gcv[v] / 25
            time_5f[v] = time_5f[v] / 25
            time_aic[v] = time_aic[v] / 25



        #Plot all results for the prediction error and runtime
        mpl.use('tkagg')
        plt.figure(figsize=(9,9))
        ax1 = plt.subplot(211)
        ax1.plot(grid_of_samples, time_mdp, 'ys--', label = r'$k^{\tau}$', linewidth=2)
        ax1.plot(grid_of_samples, time_gcv, 'r-^', label = r'$k_{GCV}$', linewidth=2)
        ax1.plot(grid_of_samples, time_aic, 'ms--', label = r'$k_{AIC}$', linewidth=2)
        ax1.plot(grid_of_samples, time_5f, 'b-+', label = r'$k_{5FCV}$', linewidth=2)
        ax1.legend(loc = 1, fontsize=9)
        if (fl_data == 0):
            ax1.set_ylim([0, 0.05])
        ax1.set_ylabel('Average time (sec)')


        ax2 = plt.subplot(212)
        ax2.plot(grid_of_samples, pred_k_mdp, 'ys--', label = r'$k^{\tau}$', linewidth=2)
        ax2.plot(grid_of_samples, pred_k_gcv, 'r-^', label = r'$k_{GCV}$', linewidth=2)
        ax2.plot(grid_of_samples, pred_k_5f, 'b-+', label = r'$k_{5FCV}$', linewidth=2)
        ax2.plot(grid_of_samples, pred_k_aic, 'ms--', label = r'$k_{AIC}$', linewidth=2)
        ax2.legend(loc = 1, fontsize=9)
        ax2.set_ylabel('Average loss')
        ax2.set_xlabel('Sub-sample size')

        plt.show()


if __name__ == "__main__":
    main()
