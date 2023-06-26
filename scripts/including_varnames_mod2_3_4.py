# -*- coding: utf-8 -*-
"""
Created on Sun Jun 25 18:49:24 2023

Descripción: en este script se busca reentrenar los modelos de DecisionTrees 
con mejores resultados para incluir los nombres de las variables a partir del 
modelo para el año 2

@author: DMatos
"""

#*****************************************************************************#
#************************** Setting up environment ***************************#
#*****************************************************************************#

#------------------------------ LIBRARIES ------------------------------------#
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import plot_confusion_matrix
from sklearn.decomposition import PCA

from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN, SMOTETomek

import os
os.chdir('C:/Users/DMatos/OneDrive - INTEC/proyecto_final/04_scripts')
#os.chdir('C:/Users/deya1/OneDrive - INTEC/proyecto_final/04_scripts')
from griding_functions import dtr_hyperparameters_tuning
from griding_functions import get_best_hyperparameters

import seaborn as sns

import matplotlib.pyplot as plt

import joblib

#------------------------------- DATA ----------------------------------------# 
path = "C:/Users/DMatos/OneDrive - INTEC/proyecto_final/"
#path = "C:/Users/deya1/OneDrive - INTEC/proyecto_final/"
dt1 = pd.read_csv(path + "03_data/02_prep/dt_model1.csv")
dt2 = pd.read_csv(path + "03_data/02_prep/dt_model2.csv")
dt3 = pd.read_csv(path + "03_data/02_prep/dt_model3.csv")
dt4 = pd.read_csv(path + "03_data/02_prep/dt_model4.csv")

#----------------------------- FUNCTIONS -------------------------------------#

def feature_importance_graph(model, train_set, graph_title):
    '''
    Description: función para graficar variables según importancia en la 
    capacidad predictiva del modelo

    Args:

    Returns:
    '''
    # get top 20 feature scores
    sorted_idx = model.feature_importances_.argsort()[::-1][:20]
    feature_names = train_set.columns[sorted_idx].to_list()
    feature_scores = model.feature_importances_[sorted_idx]
    important_features = pd.DataFrame(data=[feature_scores],
                                      columns=feature_names)

    # plot
    g = sns.barplot(data=important_features)
    g.set_title(graph_title, size=22)
    g.set_ylabel("\nFeature score\n",size=18)
    g.set_xlabel("\nFeature name\n",size=18)
    g.set_xticklabels(g.get_xticklabels(),rotation=90, 
                      horizontalalignment="center")
    g.figure.set_size_inches(30,10)
    return 

def append_prob(model, X, dataset_append, name_prob_append):
    '''
    Description: agrega al dataset de origen la probabilidad de abandono
    predicted por el modelo. Esta probabilidad será usada por el modelo para
    el siguiente año. 

    Args:

    Returns:
    '''
    prob_desert =  model.predict_proba(X)
    dataset_append[name_prob_append] = [sum(x[1:]) for x in prob_desert]
    return

def test_model(model, X_test, y_test):
    # Dict to containt tests results 
    dict_results = {}
    
    # y predicted by model
    y_test_predict =  model.predict(X_test) 

    # Score
    model_score = model.score(X_test,y_test)
    dict_results['score'] = model_score
    
    ###accuracy
    model_ac = accuracy_score(y_test, y_test_predict)
    dict_results['accuracy'] = model_ac

    ###fi1 
    model_f1 = f1_score(y_test, y_test_predict, average='weighted')
    dict_results['f1_score'] = model_f1

    ###precision
    model_p = precision_score(y_test, y_test_predict, average='weighted')
    dict_results['precision'] = model_p

    ###recall
    model_r = recall_score(y_test, y_test_predict, average='weighted')
    dict_results['recall'] = model_r

    ###confusion matrix
    plot_confusion_matrix(model, X_test, y_test)
    plt.show()

    ### funcion de costo
    prob_desert_test =  model.predict_proba(X_test)
    prob_desert= [sum(x[1:]) for x in prob_desert_test]
    sns.displot(prob_desert, kde=True)
    
    return dict_results


#*****************************************************************************#
#********************************** MODEL 1 **********************************#
#*****************************************************************************#

X1_orig = dt1[['age_in', 'year_in', 'trim_in', 'program_in_desc', 'admin_type',
               'credit_conv', 'cred_recon', 'year_birth', 'month_birth', 
               'day_birth', 'dayweek_birth', 'sex', 'nationality', 
               'birth_city', 'birth_country', 'paa_verb_fin', 'paa_mat_fin', 
               'paa_redact_fin', 'area_in_name']]

# loading saved model
m1_year1 = joblib.load(path + '05_models/decision_tree_year1.joblib')

# Appending probabilities desertion a dt1
append_prob(m1_year1, X1_orig, dt1, 'prob_m1_year1')


#*****************************************************************************#
#********************************** MODEL 2 **********************************#
#*****************************************************************************#

# Recuperando probabilidades modelo 1 
# usare las probabilidades arrojadas por el modelo con los mejores resultados
dt2 = dt2.merge(dt1[['prob_m1_year1', 'id']], how='left', on='id')

# Import dataset
X2_orig = dt2.drop(columns=['y', 'id', 'y_m2'])
y2_orig = dt2["y"]
y2_m2_orig = dt2["y_m2"]

# PCA para reducir vars courses # Estaremos usando PCA para cursos de frente 
pca = PCA(n_components = 125, random_state=42)
X2_courses = X2_orig[[x for x in X2_orig if 'course_' in x and '_quant'\
                      not in x]]
X2_train_pca = pd.DataFrame(pca.fit_transform(X2_courses), columns=None)
X2_train_pca = X2_train_pca.add_prefix('pca_') #arreglando nombre cols
X2_orig.drop(columns = X2_courses.columns, inplace=True)
X2_orig = pd.concat([X2_orig, X2_train_pca], axis=1)
explained_variance = pca.explained_variance_ratio_.sum() #para seleccionar el n_components nos basamos en que este valor estuviera por encima de 0.95


# Split data into training and test
X2_orig_train, X2_orig_test, y2_orig_train, y2_orig_test, y2_m2_orig_train, \
    y2_m2_orig_test = train_test_split(X2_orig, y2_orig, y2_m2_orig, 
                                       test_size=0.25, random_state=42)

# PROB MOD 1 VS ALL VARS MOD 1 ************************************************
# con modelo mejor resultado #m2_m2_smote 

# Dataset balanceado SMOTE all vars -------------------------------------------

X2_all_vars = dt2.merge(dt1[['age_in', 'year_in', 'trim_in', 'program_in_desc', 
                             'admin_type', 'credit_conv', 'cred_recon', 
                             'year_birth', 'month_birth', 'day_birth', 
                             'dayweek_birth', 'sex', 'nationality', 
                             'birth_city', 'birth_country', 'paa_verb_fin', 
                             'paa_mat_fin', 'paa_redact_fin', 'area_in_name', 
                             'id']], how='left', on='id').drop(columns=['y', 
                             'id', 'y_m2', 'prob_m1_year1'])

# PCA para reducir vars courses # Estaremos usando PCA para cursos de frente 
pca = PCA(n_components = 125, random_state=42)
X2_courses = X2_all_vars[[x for x in X2_all_vars if 'course_' in x and \
                          '_quant' not in x]]
X2_train_pca = pd.DataFrame(pca.fit_transform(X2_courses), columns=None)
X2_train_pca = X2_train_pca.add_prefix('pca_') #arreglando nombre columns
X2_all_vars.drop(columns = X2_courses.columns, inplace=True)
X2_all_vars = pd.concat([X2_all_vars, X2_train_pca], axis=1)
explained_variance = pca.explained_variance_ratio_.sum() #para seleccionar el n_components nos basamos en que este valor estuviera por encima de 0.95

# Split data into training and test
X2_all_vars_train, X2_all_vars_test, y2_m2_orig_train, \
    y2_m2_orig_test = train_test_split(X2_all_vars, y2_m2_orig, 
                                       test_size=0.25, random_state=42)

# Balancing train dataset with SMOTE
print('Original dataset shape %s' % y2_m2_orig_train.value_counts())
sm = SMOTE(random_state=42)
X2_m2_smote_train, y2_m2_smote_train = sm.fit_resample(X2_orig_train, 
                                                    y2_m2_orig_train)
print('Original dataset shape %s' % y2_m2_smote_train.value_counts())

"""
# Hyperparameter tunning 
grid_search2 = dtr_hyperparameters_tuning(X2_m2_smote_train, y2_m2_smote_train)
best_arg_m2_m2_smote, cv_res_m2_m2_smote, _ = \
    get_best_hyperparameters(grid_search2)
"""

# Recuperando hyperparametros del modelo ya entrenado
mod2 = joblib.load(path + '05_models/decision_tree_year2.joblib')
best_arg_m2_m2_smote = mod2.get_params() 

# Declare model
params_m2_m2_smote = best_arg_m2_m2_smote
m2_m2_smote = DecisionTreeClassifier(**params_m2_m2_smote)

# Train model
m2_m2_smote.fit(X2_m2_smote_train, y2_m2_smote_train)

# find the best features
title = "\n20 most important features for DTC Model 2 year 2 - smote\n"
feature_importance_graph(m2_m2_smote, X2_m2_smote_train, title)

# Resuts tests 
res_m2_m2_smote = test_model(m2_m2_smote, X2_orig_test, y2_m2_orig_test)

# Appending probabilities desertion a dt1
append_prob(m2_m2_smote, X2_orig, dt2, 'prob_m2_m2_smote')

#-----------------------------------------------------------------------------#
# Saving model with var names included
# modelo especializado 2do año con dataset balanceado SMOTE

filename = path+"05_models/decision_tree_year2_withvarnames.joblib"
joblib.dump(m2_m2_smote, filename)


#*****************************************************************************#
#********************************** MODEL 3 **********************************#
#*****************************************************************************#

# Recuperando probabilidades modelo 2
# usare las probabilidades arrojadas por el modelo con los mejores resultados
dt3 = dt3.merge(dt2[['prob_m2_m2_smote', 'id']], how='left', on='id')

# Dropping missing
dt3.dropna(inplace=True)
dt3.reset_index(inplace=True)

# Import dataset
X3_orig = dt3.drop(columns=['y', 'id', 'y_m3'])
y3_orig = dt3["y"]
y3_m3_orig = dt3["y_m3"]
 
# PCA para reducir vars courses # Estaremos usando PCA para cursos de frente 
pca = PCA(n_components = 125, random_state=42)
X3_courses = X3_orig[[x for x in X3_orig if 'course_' in x and '_quant'\
                      not in x]]
X3_train_pca = pd.DataFrame(pca.fit_transform(X3_courses), columns=None)
X3_train_pca = X3_train_pca.add_prefix('pca_') #arreglando nombre columns
X3_orig.drop(columns = X3_courses.columns, inplace=True)
X3_orig = pd.concat([X3_orig, X3_train_pca], axis=1)
explained_variance = pca.explained_variance_ratio_.sum() #para seleccionar el n_components nos basamos en que este valor estuviera por encima de 0.95


# Split data into training and test
X3_orig_train, X3_orig_test, y3_orig_train, y3_orig_test, y3_m3_orig_train, \
    y3_m3_orig_test = train_test_split(X3_orig, y3_orig, y3_m3_orig, 
                                       test_size=0.25, random_state=42)

# MODELO ESPECIALIZADO SEGUNDO AÑO ********************************************

# Dataset original ------------------------------------------------------------

"""
# Hyperparameter tunning 
grid_search3 = dtr_hyperparameters_tuning(X3_orig_train, y3_m3_orig_train)
best_arg_m3_m3, cv_res_m3_m3, _ = get_best_hyperparameters(grid_search3)
"""

# Recuperando hyperparametros del modelo ya entrenado
mod3 = joblib.load(path + '05_models/decision_tree_year3.joblib')
best_arg_m3_m3 = mod3.get_params() 


# Declare model
params_m3_m3 = best_arg_m3_m3
m3_year3 = DecisionTreeClassifier(**params_m3_m3)

# Train model
m3_year3.fit(X3_orig_train, y3_m3_orig_train)

# find the best features
title = "\n20 most important features for DTC Model 3 year 3- original\n"
feature_importance_graph(m3_year3, X3_orig_train, title)

# Resuts tests 
res_m3_year3 = test_model(m3_year3, X3_orig_test, y3_m3_orig_test)

# Appending probabilities desertion a dt1
append_prob(m3_year3, X3_orig, dt3, 'prob_m3_year3')

#-----------------------------------------------------------------------------#
# Saving model with var names included
# modelo especializado 3er año con dataset original

filename = path+"05_models/decision_tree_year3_withvarnames.joblib"
joblib.dump(m3_year3, filename)


#*****************************************************************************#
#********************************** MODEL 3 **********************************#
#*****************************************************************************#

""" 
Esta parte queda pendiente de ser implementada. Este es el modelo que menos 
información útil aporta, en parte porque ya en este punto la población objetivo
(estudiantes que abandonan) es la minoría. 
"""
