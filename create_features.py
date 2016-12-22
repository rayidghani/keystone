from __future__ import division
import pandas as pd
import numpy as np
from sklearn import preprocessing, cross_validation, metrics, tree, decomposition, svm
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression, Perceptron, SGDClassifier, OrthogonalMatchingPursuit, RandomizedLogisticRegression
from sklearn.neighbors.nearest_centroid import NearestCentroid, KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import ParameterGrid
from sklearn.metrics import *
from sklearn.preprocessing import StandardScaler
import random
import pylab as pl
import matplotlib.pyplot as plt
import time

import logging
from datetime import date, datetime, timedelta
from dateutil.relativedelta import relativedelta

#%matplotlib inline

log = logging.getLogger(__name__)

import yaml
import sqlalchemy
from os import path
import sqlalchemy.sql.expression as ex

from collate import collate





def create_features (feature_dates, entity_list, feature_defs):
    """
    This function returns a table with features for each entity

    Inputs:
    -------
    features_dates
    features_defs

    Output:
    -------
    table with all features for each entity

    """
