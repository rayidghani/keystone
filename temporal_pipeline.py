from __future__ import division
import pandas as pd
import numpy as np
from sklearn import preprocessing, cross_validation, metrics, tree, decomposition, svm
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression, Perceptron, SGDClassifier, OrthogonalMatchingPursuit, RandomizedLogisticRegression
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import ParameterGrid
from sklearn.metrics import *
from sklearn.preprocessing import StandardScaler
import random
import pylab as pl
import matplotlib.pyplot as plt
from scipy import optimize
import time

import logging
from datetime import date, datetime, timedelta
from dateutil.relativedelta import relativedelta


#%matplotlib inline



log = logging.getLogger(__name__)

def get_temporal_splits (start_time, end_time, prediction_windows, update_window):
    """
    Function that .

    Inputs:
    -------
    """

    results_df_columns = ['train_start_time','train_end_time','test_start_time','test_end_time','prediction_window','classifier', 'parameters','pat5']
    results_df = pd.DataFrame(data=np.zeros((0,len(results_df_columns))), columns=results_df_columns)

    start_time_date = datetime.strptime(start_time, '%Y-%m-%d')
    end_time_date = datetime.strptime(end_time, '%Y-%m-%d')


    for prediction_window in prediction_windows:
        test_end_time = end_time_date
        while (test_end_time >= start_time_date + 2 * relativedelta(months=+prediction_window)):
            test_start_time = test_end_time - relativedelta(months=+prediction_window)
            train_end_time = test_start_time  - relativedelta(days=+1) # minus 1 day
            train_start_time = train_end_time - relativedelta(months=+prediction_window)
            while (train_start_time >= start_time_date ):
                print train_start_time, train_end_time, test_start_time, test_end_time, prediction_window
                train_start_time -= relativedelta(months=+prediction_window)

                # call function to get data
                X_train, y_train = get_dataset(train_start_time,train_end_time)
                X_test, y_test = get_dataset(test_start_time, test_end_time)
                #X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=0)
                # build models
                results_df_columns = ['train_start_time','train_end_time','test_start_time','test_end_time','prediction_window','classifier', 'parameters','pat5']
                local_results_df = pd.DataFrame(data=np.zeros((0,len(results_df_columns))), columns=results_df_columns)
                #local_results_df.loc[len(local_results_df)] =

                local_results_df = clf_loop(models_to_run, clfs, grid, X_train, X_test, y_train, y_test)
                #print local_results_df

                # add columns with curent experiment config
                local_results_df['train_start_time'] = train_start_time
                local_results_df['train_end_time'] = train_end_time
                local_results_df['test_start_time'] = test_start_time
                local_results_df['test_end_time'] = test_end_time
                local_results_df['prediction_window'] = prediction_window

                # print local_results_df
                # put results in the larger df
                results_df = results_df.append(local_results_df)

            test_end_time -= relativedelta(months=+update_window)
    return results_df

def clf_loop(models_to_run, clfs, grid, X_train, X_test, y_train, y_test):

    columns = ['classifier', 'parameters', 'pat5']
    temp_results_df = pd.DataFrame(data=np.zeros((0,len(columns))), columns=columns)

    for index, clf in enumerate([clfs[x] for x in models_to_run]):
        parameter_values = grid[models_to_run[index]]
        for p in ParameterGrid(parameter_values):
            try:
                clf.set_params(**p)
                y_pred_probs = clf.fit(X_train, y_train).predict_proba(X_test)[:,1]
                # need to change this to store model and also get feature importances

                # threshold = np.sort(y_pred_probs)[::-1][int(.05*len(y_pred_probs))]
                # print threshold
                # print precision_at_k(y_test,y_pred_probs,.05)
                # plot_precision_recall_n(y_test,y_pred_probs,clf)
                temp_results_df.loc[len(temp_results_df)]=[x,clf,precision_at_k(y_test, y_pred_probs, .05)]
            except IndexError, e:
                print 'Error:', e
                continue

    return temp_results_df

def define_clfs_params():

    clfs = {'RF': RandomForestClassifier(n_estimators=50, n_jobs=-1),
        'ET': ExtraTreesClassifier(n_estimators=10, n_jobs=-1, criterion='entropy'),
        'AB': AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), algorithm="SAMME", n_estimators=200),
        'LR': LogisticRegression(penalty='l1', C=1e5),
        'SVM': svm.SVC(kernel='linear', probability=True, random_state=0),
        'GB': GradientBoostingClassifier(learning_rate=0.05, subsample=0.5, max_depth=6, n_estimators=10),
        'NB': GaussianNB(),
        'DT': DecisionTreeClassifier(),
        'SGD': SGDClassifier(loss="hinge", penalty="l2"),
        'KNN': KNeighborsClassifier(n_neighbors=3),
        'TESTRF': RandomForestClassifier(n_estimators=2, n_jobs=-1)
            }

    grid = {
    'RF':{'n_estimators': [1,10,100,1000,10000], 'max_depth': [1,5,10,20,50,100], 'max_features': ['sqrt','log2'],'min_samples_split': [2,5,10]},
    'LR': { 'penalty': ['l1','l2'], 'C': [0.00001,0.0001,0.001,0.01,0.1,1,10]},
    'SGD': { 'loss': ['hinge','log','perceptron'], 'penalty': ['l2','l1','elasticnet']},
    'ET': { 'n_estimators': [1,10,100,1000,10000], 'criterion' : ['gini', 'entropy'] ,'max_depth': [1,5,10,20,50,100], 'max_features': ['sqrt','log2'],'min_samples_split': [2,5,10]},
    'AB': { 'algorithm': ['SAMME', 'SAMME.R'], 'n_estimators': [1,10,100,1000,10000]},
    'GB': {'n_estimators': [1,10,100,1000,10000], 'learning_rate' : [0.001,0.01,0.05,0.1,0.5],'subsample' : [0.1,0.5,1.0], 'max_depth': [1,3,5,10,20,50,100]},
    'NB' : {},
    'DT': {'criterion': ['gini', 'entropy'], 'max_depth': [1,5,10,20,50,100], 'max_features': ['sqrt','log2'],'min_samples_split': [2,5,10]},
    'SVM' :{'C' :[0.00001,0.0001,0.001,0.01,0.1,1,10],'kernel':['linear']},
    'KNN' :{'n_neighbors': [1,5,10,25,50,100],'weights': ['uniform','distance'],'algorithm': ['auto','ball_tree','kd_tree']},
    'TESTRF':{'n_estimators': [2,5], 'max_depth': [5], 'max_features': ['sqrt'],'min_samples_split': [10]}

           }
    return clfs, grid


def precision_at_k(y_true, y_scores, k):
    threshold = np.sort(y_scores)[::-1][int(k*len(y_scores))]
    y_pred = np.asarray([1 if i >= threshold else 0 for i in y_scores])
    return metrics.precision_score(y_true, y_pred)


def get_dataset (start_time, end_time, prediction_window, feature_list, label_list):
    """
    Function that returns the dataset.

    Inputs:
    -------
     """

    # Features:
    # assume a table with pre-generated features of the following structure
    # entity_id, feature_date, feature_1, feature_2, feature_3, ...
    # assume a table with outcomes that can be used as labels
    # entity_id, event_time, outcome_time, outcome_type, outcome_value

    query = ("SELECT {}, {}, {} as label"
                "FROM features.{} f"
                "LEFT JOIN  outcomes.{} o on f.{} = o.{}"
                "AND outcome_timestamp =< feature_date + prediction_window"
                "AND outcome_timestamp > feature_date"
                "WHERE o.outcome_type IN label_list"
                "AND feature_date >= '{}' AND feature_date <= '{}'"

                .format(
                    id_column,
                    feature_list,
                    outcome_value_column,
                    feature_table,
                    outcomes_table,
                    id_column,
                    id_column,
                    start_time,
                    end_time))

    """
    select entity_id, features, outcome_value from feature_table f left join
    outcomes_table o on f.entity_id = o.entity_id
    and outcome_timestamp =< feature_date + prediction_window
    and outcome_timestamp > feature_date
    where o.outcome_type = label
    and feature_date >= train_start_time
    and feature_date <= train_end_time
    """
    all_data = pd.read_sql(query, con=db_conn)
    all_data = all_data.set_index(id_column)
    return all_data[features], all_data.label
