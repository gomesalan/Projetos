import pandas as pd
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt

## Imputar valores 
from sklearn.impute import SimpleImputer

## Codificação numérica as variáveis qualitativas
from sklearn.preprocessing import OneHotEncoder

## Normalização/Padronização 
from sklearn.preprocessing import MinMaxScaler, StandardScaler

## Balanceamento de classes 
from imblearn.over_sampling import SMOTE 

## Divisão treino/teste 
from sklearn.model_selection import train_test_split
# Cross Validation 
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate

## Pipelines 
from imblearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer, make_column_transformer 

## Tratamento de outliers 
from sklearn.neighbors import LocalOutlierFactor

## Métricas de avaliação. 
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score 
from sklearn.metrics import confusion_matrix

## Algoritmos de aprendizagem. 
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier)

## Otimização 
from sklearn.model_selection import GridSearchCV

import warnings
warnings.filterwarnings("ignore")