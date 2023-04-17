# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 00:24:30 2023

@author: DMatos
"""

import pandas as pd
import os
import json
import warnings
warnings.filterwarnings('ignore')


# Import specific ML-purpose libraries (sklearn)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import LinearSVR
from sklearn.ensemble import RandomForestClassifier
# Reference: https://scikit-learn.org/stable/modules/neural_networks_supervised.html#tips-on-practical-use
from sklearn.pipeline import Pipeline
# References: https://machinelearningmastery.com/standardscaler-and-minmaxscaler-transforms-in-python/
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.tree import DecisionTreeClassifier
# BayesSearchCV usage: https://machinelearningmastery.com/scikit-optimize-for-hyperparameter-tuning-in-machine-learning/
# Grid search: https://scikit-learn.org/stable/modules/grid_search.html
from skopt import BayesSearchCV

# For using HalvingGridSearchCV, it is necessary to use the following imports:
# Source: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.HalvingGridSearchCV.html#sklearn-model-selection-halvinggridsearchcv
# explicitly require this experimental feature
from sklearn.experimental import enable_halving_search_cv # noqa
# now you can import normally from model_selection
from sklearn.model_selection import HalvingGridSearchCV







'''DesicionTreeRegressor hyperparameters grinding using BayesSearchCV method
------------------------------------------------------------
'''
def dtr_hyperparameters_tuning(X_train: pd.DataFrame, y_train: pd.DataFrame) -> BayesSearchCV:
    '''This function takes as input a couple of DataFrames that will be used for training a DesicionTreeRegressor as many times as wanted, in order to find the best hyperparameters. 

    Args:
        - X_train (`pd.DataFrame`): The training set of features. The shape must be (m, n), where 'm' is the number of samples, and 'n' is the number of features. 
        - y_train (`pd.DataFrame`): The training set of ground-truth values. The shape must be (m, 1), where 'm' is the number of ground-truth values that corresponds to each sample

    Returns:
        `BayesSearchCV`: The set of trained models and statistics about traininig behavior, so one can choose the best model.
    '''
    # Parameter selection for DesicionTreeRegressor: 
    # https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html
    grid = {
        'criterion': ["gini", "entropy"],
        'splitter': ['best'],
        'max_depth': [None],
        'min_samples_split': [2,3,4,5],
        'min_samples_leaf': [2,3,4,5],
        'min_weight_fraction_leaf': [0.0,0.01,0.02,0.03],
        'max_features': [None],
        'random_state': [42],
        'max_leaf_nodes': [None],
        'min_impurity_decrease': [0.0,0.01,0.02,0.04],
        'ccp_alpha': [0.02,0.03]
    }

    # BayesSearchCV configuration: 
    # https://scikit-optimize.github.io/stable/modules/generated/skopt.BayesSearchCV.html
    print("LLEGUE antes de Bayes")
    grid_search = BayesSearchCV(
        estimator = DecisionTreeClassifier(),
        search_spaces = grid,
        n_iter = 150,
        # Check carefully
        # https://scikit-learn.org/stable/modules/model_evaluation.html#the-scoring-parameter-defining-model-evaluation-rules
        scoring='accuracy',
        cv = 2,
        n_jobs = 1,
        verbose = 1
    )
    print("LLEGUE despues de Bayes")
    # Train each resultant combination
    grid_search.fit(X_train, y_train)
    return grid_search




'''RandomForestRegressor hyperparameters grinding using HalvingGridSearchCV method
------------------------------------------------------------
'''
# Parameter optimization via HalvingGridSearchCV for RandomForestRegressor
def rfr_hyperparameters_tuning(X_train: pd.DataFrame, y_train: pd.DataFrame) -> HalvingGridSearchCV:
    '''This function takes as input a couple of DataFrames that will be used for training a RandomForestRegressor, in order to find the best hyperparameters. 

    Args:
        - X_train (`pd.DataFrame`): The training set of features. The shape must be (m, n), where 'm' is the number of samples, and 'n' is the number of features. 
        - y_train (`pd.DataFrame`): The training set of ground-truth values. The shape must be (m, 1), where 'm' is the number of ground-truth values that corresponds to each sample

    Returns:
        `HalvingGridSearchCV`: The set of trained models and statistics about traininig behavior, so one can choose the best model.
    '''
    # Parameter selection for RandomForestRegressor: 
    # https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html
    grid = {
        # 'n_estimators': [100, 30, 80, 120, 200, 250],
        'n_estimators': [200, 300],
        # 'criterion': ["squared_error", "friedman_mse"],
        'criterion': ["gini"],
        # 'max_depth': [None, 3, 6, 9, 12],
        'max_depth': [150, 100],
        #'min_samples_split': [2, 3, 0.01, 0.08],
        'min_samples_split': [2],
        # 'min_samples_leaf': [1, 2, 3, 0.001],
        'min_samples_leaf': [1],
        # 'max_features': [1.0, 0.5, 0.7, 0.9],
        'max_features': [0.35],
        # 'max_leaf_nodes': [None, 50, 100, 200],
        'max_leaf_nodes': [None],
        # 'bootstrap': [True],
        # 'oob_score': [False],
        'random_state': [42],
        # The following selection is being taken as a reference from:
        # https://scikit-learn.org/stable/auto_examples/tree/plot_cost_complexity_pruning.html#accuracy-vs-alpha-for-training-and-testing-sets
        # 'ccp_alpha': [0.0, 0.01, 0.015, 0.03, 0.04],
        'ccp_alpha': [0.05],
        # 'max_samples': [None, 0.7, 0.9]
        'max_samples': [None]
    }

    # HalvingGridSearchCV configuration: 
    # https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.HalvingGridSearchCV.html#sklearn-model-selection-halvinggridsearchcv
    grid_search = HalvingGridSearchCV(
        estimator = RandomForestClassifier(),
        param_grid = grid,
        factor = 2,
        scoring='accuracy',
        cv = 7,
        n_jobs = 1,
        verbose = 1
    )

    # Train each resultant combination
    grid_search.fit(X_train, y_train)
    return grid_search  





'''Getting hyperparameters from the grid object
------------------------------------------------------------------------
'''
# Construct a DataFrame from the results obtained after applying a GrindCV method.
def get_best_hyperparameters(grid_search: BayesSearchCV) -> tuple[dict, pd.DataFrame, float]:
    '''This function constructs a DataFrame from the results obtained after applying a Hyperparameter Tuning (BayesGrindCV, for example)

    Args:
        - grid_search (`GrindCV method`): The Hyperparameter Tuning made beforehand

    Returns:
        `tuple[dict, pd.DataFrame, float]`: A tuple containing a dictionary of the best parameters, a DataFrame that contains the entire results, and the best MAE associated with the best parameters.
    '''
    cv_results = pd.DataFrame(grid_search.cv_results_)

    # Since the method for applying a desired scoring function (for this case, mean_absolute_error) is not possible for this method (that is because it is implemented for MAXIMIZING the "mean_test_score" attribute), let us implement the negative mean_absolute_error. 
    # That is, set the "scoring" parameter as "neg_mean_absolute_error" (you can get all of the available metrics using sklearn.metrics.get_scorer_names() method, which will give an array of available callers (in string) ).

    # Next, implement the absolute value into the dataframe["mean_test_score"] column.
    cv_results["mean_test_score"] = cv_results["mean_test_score"].apply(lambda x: abs(x))
    # Order the dataframe in descending order, since the first one will be the "best" model result. Also, reset the entire index
    cv_results.sort_values(by=["mean_test_score"], ignore_index=True, inplace=True, ascending=False)
    # Get all of the rows which "rank_test_score" column equals 1
    cv_results_bests = cv_results[cv_results["rank_test_score"] == 1]
    best_row_with_less_duration = cv_results_bests[cv_results_bests["mean_fit_time"] == cv_results_bests["mean_fit_time"].min()].reset_index()
    # Pick up the first "params" value from the first row, since this is the best result
    estimated_MAE_value = best_row_with_less_duration.loc[0,"mean_test_score"]
    best_arguments = dict(best_row_with_less_duration.at[0, "params"])
    return (best_arguments, cv_results, estimated_MAE_value)




