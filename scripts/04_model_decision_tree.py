# -*- coding: utf-8 -*-
"""
Created on Sat Jan 28 22:56:14 2023

Script for developing Decision Trees for all models

Mejoras implementadas de frente en este modelado a partir de los resultados 
del primer modelado: 
    - 2 clases en lugar de 3 
    - Aplicar PCA de frente para course vars modelos >= 2

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
from griding_functions import dtr_hyperparameters_tuning
from griding_functions import get_best_hyperparameters

import seaborn as sns

import matplotlib.pyplot as plt

import joblib


"""
#*WORKING DIRECTORY*#
import os
cwd = os.getcwd() #for getting current working directory
os.chdir('/tmp') #for changing working directory 
"""

#------------------------------- DATA ----------------------------------------# 
path = "C:/Users/DMatos/OneDrive - INTEC/proyecto_final/"
dt1 = pd.read_csv(path + "03_data/02_prep/dt_model1.csv")
dt2 = pd.read_csv(path + "03_data/02_prep/dt_model2.csv")
dt3 = pd.read_csv(path + "03_data/02_prep/dt_model3.csv")
dt4 = pd.read_csv(path + "03_data/02_prep/dt_model4.csv")

#----------------------------- FUNCITONS -------------------------------------#

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

# Modelos generales vs. especializados por año ()

# Modelo general **************************************************************

# Dataset original vs. dataset balanceado (de aqui escojo 1 que me convenga)

# Dataset original ------------------------------------------------------------

X1_orig = dt1[['age_in', 'year_in', 'trim_in', 'program_in_desc', 'admin_type',
               'credit_conv', 'cred_recon', 'year_birth', 'month_birth', 
               'day_birth', 'dayweek_birth', 'sex', 'nationality', 
               'birth_city', 'birth_country', 'paa_verb_fin', 'paa_mat_fin', 
               'paa_redact_fin', 'area_in_name']]
y1_orig = dt1["y"]
y1_m1_orig = dt1["y_m1"]

# Split data into training and test
X1_orig_train, X1_orig_test, y1_orig_train, y1_orig_test, y1_m1_orig_train, \
    y1_m1_orig_test = train_test_split(X1_orig, y1_orig, y1_m1_orig, 
                                       test_size=0.25, random_state=42)

# Hyperparameter tunning 
grid_search = dtr_hyperparameters_tuning(X1_orig_train, y1_orig_train)
best_arg_m1_orig, cv_res_m1_orig, _ = get_best_hyperparameters(grid_search)

# Declare model 
params_m1_orig = best_arg_m1_orig
m1_orig = DecisionTreeClassifier(**params_m1_orig)

# Train model
m1_orig.fit(X1_orig_train, y1_orig_train)

# find the best features
title = "\n20 most important features for DTC Model 1 - original\n"
feature_importance_graph(m1_orig, X1_orig_train, title)

# Resuts tests 
res_m1_orig = test_model(m1_orig, X1_orig_test, y1_orig_test)

# Appending probabilities desertion a dt1
append_prob(m1_orig, X1_orig, dt1, 'prob_m1_orig')


# Dataset balanced SMOTE ------------------------------------------------------

# Balancing train dataset with SMOTE
print('Original dataset shape %s' % y1_orig_train.value_counts())
sm = SMOTE(random_state=42)
X1_smote_train, y1_smote_train = sm.fit_resample(X1_orig_train, y1_orig_train)
print('Original dataset shape %s' % y1_smote_train.value_counts())

# Hyperparameter tunning 
grid_search = dtr_hyperparameters_tuning(X1_smote_train, y1_smote_train)
best_arg_m1_smote, cv_res_m1_smote, _ = get_best_hyperparameters(grid_search)

# Declare model 
params_m1_smote = best_arg_m1_smote
m1_smote = DecisionTreeClassifier(**params_m1_smote)

# Train model
m1_smote.fit(X1_smote_train, y1_smote_train)

# find the best features
title = "\n20 most important features for DTC Model 1 - SMOTE\n"
feature_importance_graph(m1_smote, X1_smote_train, title)

# Resuts tests 
res_m1_smote = test_model(m1_smote, X1_orig_test, y1_orig_test)

# Appending probabilities desertion a dt1
append_prob(m1_smote, X1_orig, dt1, 'prob_m1_smote')


# Dataset balanced SMOTE-ENN --------------------------------------------------

# Balancing train dataset with SMOTE-ENN
print('Original dataset shape %s' % y1_orig_train.value_counts())
smenn = SMOTEENN(random_state=42)
X1_smoteenn_train, y1_smoteenn_train = sm.fit_resample(X1_orig_train, 
                                                       y1_orig_train)
print('Original dataset shape %s' % y1_smoteenn_train.value_counts())

# Hyperparameter tunning 
grid_search = dtr_hyperparameters_tuning(X1_smoteenn_train, y1_smoteenn_train)
best_arg_m1_smoteenn, cv_res_m1_smoteenn, _ = \
    get_best_hyperparameters(grid_search)

# Declare model 
params_m1_smoteenn = best_arg_m1_smoteenn
m1_smoteenn = DecisionTreeClassifier(**params_m1_smoteenn)

# Train model
m1_smoteenn.fit(X1_smoteenn_train, y1_smoteenn_train)

# find the best features
title = "\n20 most important features for DTC Model 1 - SMOTE-ENN\n"
feature_importance_graph(m1_smoteenn, X1_smoteenn_train, title)

# Resuts tests 
res_m1_smoteenn = test_model(m1_smoteenn, X1_orig_test, y1_orig_test)

# Appending probabilities desertion a dt1
append_prob(m1_smoteenn, X1_orig, dt1, 'prob_m1_smoteenn')


# Dataset balanced SMOTE-Tomek ------------------------------------------------

# Balancing train dataset with SMOTETomek
print('Original dataset shape %s' % y1_orig_train.value_counts())
smtmk = SMOTETomek(random_state=42)
X1_smotetmk_train, y1_smotetmk_train = sm.fit_resample(X1_orig_train, 
                                                       y1_orig_train)
print('Original dataset shape %s' % y1_smotetmk_train.value_counts())

# Hyperparameter tunning 
grid_search = dtr_hyperparameters_tuning(X1_smotetmk_train, y1_smotetmk_train)
best_arg_m1_smotetmk, cv_res_m1_smotetmk, _ = \
    get_best_hyperparameters(grid_search)

# Declare model 
params_m1_smotetmk = best_arg_m1_smotetmk
m1_smotetmk = DecisionTreeClassifier(**params_m1_smotetmk)

# Train model
m1_smotetmk.fit(X1_smotetmk_train, y1_smotetmk_train)

# find the best features
title = "\n20 most important features for DTC Model 1 - SMOTE-Tomek\n"
feature_importance_graph(m1_smotetmk, X1_smotetmk_train, title)

# Resuts tests 
res_m1_smotetmk = test_model(m1_smotetmk, X1_orig_test, y1_orig_test)

# Appending probabilities desertion a dt1
append_prob(m1_smotetmk, X1_orig, dt1, 'prob_m1_smoteenn')


# MODELO ESPECIALIZADO PRIMER AÑO *********************************************
## el que mejor se comporte de los anteriores se compara con y_m1 #SMOTE

# Dataset balanced SMOTE ------------------------------------------------------

# Balancing train dataset with SMOTE
print('Original dataset shape %s' % y1_m1_orig_train.value_counts())
sm = SMOTE(random_state=42)
X1_m1_smote_train, y1_m1_smote_train = sm.fit_resample(X1_orig_train, 
                                                       y1_m1_orig_train)
print('Original dataset shape %s' % y1_m1_smote_train.value_counts())

# Hyperparameter tunning 
grid_search = dtr_hyperparameters_tuning(X1_m1_smote_train, y1_m1_smote_train)
best_arg_m1_m1_smote, cv_res_m1_m1_smote, _ = \
    get_best_hyperparameters(grid_search)

# Declare model 
params_m1_m1_smote = best_arg_m1_smote
m1_m1_smote = DecisionTreeClassifier(**params_m1_m1_smote)

# Train model
m1_m1_smote.fit(X1_m1_smote_train, y1_m1_smote_train)

# find the best features
title = "\n20 most important features for DTC Model 1 year 1 - SMOTE\n"
feature_importance_graph(m1_m1_smote, X1_m1_smote_train, title)

# Resuts tests 
res_m1_m1_smote = test_model(m1_m1_smote, X1_orig_test, y1_m1_orig_test)

# Appending probabilities desertion a dt1
append_prob(m1_m1_smote, X1_orig, dt1, 'prob_m1_m1_smote')

# Dataset original ------------------------------------------------------------

## estimar este mismo sin SMOTE

# Hyperparameter tunning 
grid_search = dtr_hyperparameters_tuning(X1_orig_train, y1_m1_orig_train)
best_arg_m1_m1, cv_res_m1_m1, _ = get_best_hyperparameters(grid_search)

# Declare model 
params_m1_m1 = best_arg_m1_m1
m1_year1 = DecisionTreeClassifier(**params_m1_m1)

# Train model
m1_year1.fit(X1_orig_train, y1_m1_orig_train)

# find the best features
title = "\n20 most important features for DTC Model 1 year 1\n"
feature_importance_graph(m1_year1, X1_orig_train, title)

# Resuts tests 
res_m1_year1 = test_model(m1_year1, X1_orig_test, y1_m1_orig_test)

# Appending probabilities desertion a dt1
append_prob(m1_year1, X1_orig, dt1, 'prob_m1_year1')

#TODO evaluar sin incluir year_in

#-----------------------------------------------------------------------------#
# Saving model with best performance 
# modelo especializado 1er año con dataset original

filename = path+"05_models/decision_tree_year1.joblib"
joblib.dump(m1_year1, filename)

#md_ex = joblib.load(filename)
#test_model(md_ex, X1_orig_test, y1_m1_orig_test)

#*****************************************************************************#
#********************************** MODEL 2 **********************************#
#*****************************************************************************#

# Recuperando probabilidades modelo 1 
# usare las probabilidades arrojadas por el modelo con los mejores resultados
dt2 = dt2.merge(dt1[['prob_m1_year1', 'id']], how='left', on='id')

# Import dataset
X2_orig = dt2.drop(columns=['y', 'id', 'y_m2', 'prob_m1_orig'])
y2_orig = dt2["y"]
y2_m2_orig = dt2["y_m2"]
 
# PCA para reducir vars courses # Estaremos usando PCA para cursos de frente 
pca = PCA(n_components = 125, random_state=42)
X2_courses = X2_orig[[x for x in X2_orig if 'course_' in x and '_quant'\
                      not in x]]
X2_train_pca = pd.DataFrame(pca.fit_transform(X2_courses), columns=None)
X2_orig.drop(columns = X2_courses.columns, inplace=True)
X2_orig = pd.concat([X2_orig, X2_train_pca], axis=1)
explained_variance = pca.explained_variance_ratio_.sum() #para seleccionar el n_components nos basamos en que este valor estuviera por encima de 0.95


# Split data into training and test
X2_orig_train, X2_orig_test, y2_orig_train, y2_orig_test, y2_m2_orig_train, \
    y2_m2_orig_test = train_test_split(X2_orig, y2_orig, y2_m2_orig, 
                                       test_size=0.25, random_state=42)
    
# Modelos generales vs. especializados por año 
# MODELO GENERAL **************************************************************

# Dataset original vs. dataset balanceado (de aqui escojo 1 que me convenga) 
#SMOTE

# Dataset original ------------------------------------------------------------

# Hyperparameter tunning 
grid_search2 = dtr_hyperparameters_tuning(X2_orig_train, y2_orig_train)
best_arg_m2_orig, cv_res_m2_orig, _ = get_best_hyperparameters(grid_search2)

# Declare model
params_m2_orig = best_arg_m2_orig
m2_orig = DecisionTreeClassifier(**params_m2_orig)

# Train model
m2_orig.fit(X2_orig_train, y2_orig_train)

# find the best features
title = "\n20 most important features for DTC Model 2 - original\n"
feature_importance_graph(m2_orig, X2_orig_train, title)

# Resuts tests 
res_m2_orig = test_model(m2_orig, X2_orig_test, y2_orig_test)

# Appending probabilities desertion a dt1
append_prob(m2_orig, X2_orig, dt2, 'prob_m2_orig')

# Dataset SMOTE ---------------------------------------------------------------

# Balancing train dataset with SMOTE
print('Original dataset shape %s' % y2_orig_train.value_counts())
sm = SMOTE(random_state=42)
X2_smote_train, y2_smote_train = sm.fit_resample(X2_orig_train, y2_orig_train)
print('Original dataset shape %s' % y2_smote_train.value_counts())

# Hyperparameter tunning 
grid_search2 = dtr_hyperparameters_tuning(X2_smote_train, y2_smote_train)
best_arg_m2_smote, cv_res_m2_smote, _ = get_best_hyperparameters(grid_search2)

# Declare model
params_m2_smote = best_arg_m2_smote
m2_smote = DecisionTreeClassifier(**params_m2_smote)

# Train model
m2_smote.fit(X2_smote_train, y2_smote_train)

# find the best features
title = "\n20 most important features for DTC Model 2 - original\n"
feature_importance_graph(m2_smote, X2_smote_train, title)

# Resuts tests 
res_m2_smote = test_model(m2_smote, X2_orig_test, y2_orig_test)

# Appending probabilities desertion a dt1
append_prob(m2_smote, X2_orig, dt2, 'prob_m2_smote')


# MODELO ESPECIALIZADO SEGUNDO AÑO ********************************************

# Dataset original ------------------------------------------------------------

# Hyperparameter tunning 
grid_search2 = dtr_hyperparameters_tuning(X2_orig_train, y2_m2_orig_train)
best_arg_m2_m2, cv_res_m2_m2, _ = get_best_hyperparameters(grid_search2)

# Declare model
params_m2_m2 = best_arg_m2_m2
m2_year2 = DecisionTreeClassifier(**params_m2_m2)

# Train model
m2_year2.fit(X2_orig_train, y2_m2_orig_train)

# find the best features
title = "\n20 most important features for DTC Model 2 year 2- original\n"
feature_importance_graph(m2_year2, X2_orig_train, title)

# Resuts tests 
res_m2_year2 = test_model(m2_year2, X2_orig_test, y2_m2_orig_test)

# Appending probabilities desertion a dt1
append_prob(m2_year2, X2_orig, dt2, 'prob_m2_year2')

# Dataset SMOTE ---------------------------------------------------------------

# Balancing train dataset with SMOTE
print('Original dataset shape %s' % y2_m2_orig_train.value_counts())
sm = SMOTE(random_state=42)
X2_m2_smote_train, y2_m2_smote_train = sm.fit_resample(X2_orig_train, 
                                                    y2_m2_orig_train)
print('Original dataset shape %s' % y2_m2_smote_train.value_counts())

# Hyperparameter tunning 
grid_search2 = dtr_hyperparameters_tuning(X2_m2_smote_train, y2_m2_smote_train)
best_arg_m2_m2_smote, cv_res_m2_m2_smote, _ = \
    get_best_hyperparameters(grid_search2)

# Declare model
params_m2_m2_smote = best_arg_m2_m2_smote
m2_m2_smote = DecisionTreeClassifier(**params_m2_smote)

# Train model
m2_m2_smote.fit(X2_m2_smote_train, y2_m2_smote_train)

# find the best features
title = "\n20 most important features for DTC Model 2 year 2 - smote\n"
feature_importance_graph(m2_m2_smote, X2_m2_smote_train, title)

# Resuts tests 
res_m2_m2_smote = test_model(m2_m2_smote, X2_orig_test, y2_m2_orig_test)

# Appending probabilities desertion a dt1
append_prob(m2_m2_smote, X2_orig, dt2, 'prob_m2_m2_smote')


# Otput modelo anterior vs. all variables in prev year 
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

# Hyperparameter tunning 
grid_search2 = dtr_hyperparameters_tuning(X2_m2_smote_train, y2_m2_smote_train)
best_arg_m2_m2_smote, cv_res_m2_m2_smote, _ = \
    get_best_hyperparameters(grid_search2)

# Declare model
params_m2_m2_smote = best_arg_m2_m2_smote
m2_m2_smote = DecisionTreeClassifier(**params_m2_smote)

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
# Saving model with best performance 
# modelo especializado 2do año con dataset balanceado SMOTE

filename = path+"05_models/decision_tree_year2.joblib"
joblib.dump(m2_m2_smote, filename)


#*****************************************************************************#
#********************************** MODEL 3 **********************************#
#*****************************************************************************#

# Dataset original vs. dataset balanceado (de aqui escojo 1 que me convenga)

# Modelos generales vs. especializados por año 

# Otput modelo anterior vs. all variables in prev year 


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
X3_orig.drop(columns = X3_courses.columns, inplace=True)
X3_orig = pd.concat([X3_orig, X3_train_pca], axis=1)
explained_variance = pca.explained_variance_ratio_.sum() #para seleccionar el n_components nos basamos en que este valor estuviera por encima de 0.95


# Split data into training and test
X3_orig_train, X3_orig_test, y3_orig_train, y3_orig_test, y3_m3_orig_train, \
    y3_m3_orig_test = train_test_split(X3_orig, y3_orig, y3_m3_orig, 
                                       test_size=0.25, random_state=42)

    
# Modelos generales vs. especializados por año 
# MODELO GENERAL **************************************************************

# Dataset original vs. dataset balanceado (de aqui escojo 1 que me convenga) 
#SMOTE

# Dataset original ------------------------------------------------------------

# Hyperparameter tunning 
grid_search3 = dtr_hyperparameters_tuning(X3_orig_train, y3_orig_train)
best_arg_m3_orig, cv_res_m3_orig, _ = get_best_hyperparameters(grid_search3)

# Declare model
params_m3_orig = best_arg_m3_orig
m3_orig = DecisionTreeClassifier(**params_m3_orig)

# Train model
m3_orig.fit(X3_orig_train, y3_orig_train)

# find the best features
title = "\n20 most important features for DTC Model 3 - original\n"
feature_importance_graph(m3_orig, X3_orig_train, title)

# Resuts tests 
res_m3_orig = test_model(m3_orig, X3_orig_test, y3_orig_test)

# Appending probabilities desertion a dt1
append_prob(m3_orig, X3_orig, dt3, 'prob_m3_orig')

# Dataset SMOTE ---------------------------------------------------------------

# Balancing train dataset with SMOTE
print('Original dataset shape %s' % y3_orig_train.value_counts())
sm = SMOTE(random_state=42)
X3_smote_train, y3_smote_train = sm.fit_resample(X3_orig_train, y3_orig_train)
print('Original dataset shape %s' % y3_smote_train.value_counts())

# Hyperparameter tunning 
grid_search3 = dtr_hyperparameters_tuning(X3_smote_train, y3_smote_train)
best_arg_m3_smote, cv_res_m3_smote, _ = get_best_hyperparameters(grid_search3)

# Declare model
params_m3_smote = best_arg_m3_smote
m3_smote = DecisionTreeClassifier(**params_m3_smote)

# Train model
m3_smote.fit(X3_smote_train, y3_smote_train)

# find the best features
title = "\n20 most important features for DTC Model 3 - original\n"
feature_importance_graph(m3_smote, X3_smote_train, title)

# Resuts tests 
res_m3_smote = test_model(m3_smote, X3_orig_test, y3_orig_test)

# Appending probabilities desertion a dt3
append_prob(m3_smote, X3_orig, dt3, 'prob_m3_smote')


# MODELO ESPECIALIZADO SEGUNDO AÑO ********************************************

# Dataset original ------------------------------------------------------------

# Hyperparameter tunning 
grid_search3 = dtr_hyperparameters_tuning(X3_orig_train, y3_m3_orig_train)
best_arg_m3_m3, cv_res_m3_m3, _ = get_best_hyperparameters(grid_search3)

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

# Dataset SMOTE ---------------------------------------------------------------

# Balancing train dataset with SMOTE
print('Original dataset shape %s' % y3_m3_orig_train.value_counts())
sm = SMOTE(random_state=42)
X3_m3_smote_train, y3_m3_smote_train = sm.fit_resample(X3_orig_train, 
                                                    y3_m3_orig_train)
print('Original dataset shape %s' % y3_m3_smote_train.value_counts())

# Hyperparameter tunning 
grid_search3 = dtr_hyperparameters_tuning(X3_m3_smote_train, y3_m3_smote_train)
best_arg_m3_m3_smote, cv_res_m3_m3_smote, _ = \
    get_best_hyperparameters(grid_search3)

# Declare model
params_m3_m3_smote = best_arg_m3_m3_smote
m3_m3_smote = DecisionTreeClassifier(**params_m3_smote)

# Train model
m3_m3_smote.fit(X3_m3_smote_train, y3_m3_smote_train)

# find the best features
title = "\n20 most important features for DTC Model 3 year 3 - smote\n"
feature_importance_graph(m3_m3_smote, X3_m3_smote_train, title)

# Resuts tests 
res_m3_m3_smote = test_model(m3_m3_smote, X3_orig_test, y3_m3_orig_test)

# Appending probabilities desertion a dt1
append_prob(m3_m3_smote, X3_orig, dt3, 'prob_m3_m3_smote')

#-----------------------------------------------------------------------------#
# Saving model with best performance 
# modelo especializado 3er año con dataset original

filename = path+"05_models/decision_tree_year3.joblib"
joblib.dump(m3_year3, filename)


#*****************************************************************************#
#********************************** MODEL 4 **********************************#
#*****************************************************************************#

# Dataset original vs. dataset balanceado (de aqui escojo 1 que me convenga)

# Modelos generales vs. especializados por año 

# Otput modelo anterior vs. all variables in prev year 

# Recuperando probabilidades modelo 3
# usare las probabilidades arrojadas por el modelo con los mejores resultados
dt4 = dt4.merge(dt3[['prob_m3_year3', 'id']], how='left', on='id') 

# Dropping missing values 
dt4.dropna(inplace=True)
dt4.reset_index(inplace=True)

# Import dataset
X4_orig = dt4.drop(columns=['y', 'id', 'y_m4'])
y4_orig = dt4["y"]
y4_m4_orig = dt4["y_m4"]
 
# PCA para reducir vars courses # Estaremos usando PCA para cursos de frente 
pca = PCA(n_components = 125, random_state=42)
X4_courses = X4_orig[[x for x in X4_orig if 'course_' in x and '_quant'\
                      not in x]]
X4_train_pca = pd.DataFrame(pca.fit_transform(X4_courses), columns=None)
X4_orig.drop(columns = X4_courses.columns, inplace=True)
X4_orig = pd.concat([X4_orig, X4_train_pca], axis=1)
explained_variance = pca.explained_variance_ratio_.sum() #para seleccionar el n_components nos basamos en que este valor estuviera por encima de 0.95


# Split data into training and test
X4_orig_train, X4_orig_test, y4_orig_train, y4_orig_test, y4_m4_orig_train, \
    y4_m4_orig_test = train_test_split(X4_orig, y4_orig, y4_m4_orig, 
                                       test_size=0.25, random_state=42)
    
# Modelos generales vs. especializados por año 
# MODELO GENERAL **************************************************************

# Dataset original vs. dataset balanceado (de aqui escojo 1 que me convenga) 
#SMOTE

# Dataset original ------------------------------------------------------------

# Hyperparameter tunning 
grid_search4 = dtr_hyperparameters_tuning(X4_orig_train, y4_orig_train)
best_arg_m4_orig, cv_res_m4_orig, _ = get_best_hyperparameters(grid_search4)

# Declare model
params_m4_orig = best_arg_m4_orig
m4_orig = DecisionTreeClassifier(**params_m4_orig)

# Train model
m4_orig.fit(X4_orig_train, y4_orig_train)

# find the best features
title = "\n20 most important features for DTC Model 4 - original\n"
feature_importance_graph(m4_orig, X4_orig_train, title)

# Resuts tests 
res_m4_orig = test_model(m4_orig, X4_orig_test, y4_orig_test)

# Appending probabilities desertion a dt4
append_prob(m4_orig, X4_orig, dt4, 'prob_m4_orig')

# Dataset SMOTE ---------------------------------------------------------------

# Balancing train dataset with SMOTE
print('Original dataset shape %s' % y4_orig_train.value_counts())
sm = SMOTE(random_state=42)
X4_smote_train, y4_smote_train = sm.fit_resample(X4_orig_train, y4_orig_train)
print('Original dataset shape %s' % y4_smote_train.value_counts())

# Hyperparameter tunning 
grid_search4 = dtr_hyperparameters_tuning(X4_smote_train, y4_smote_train)
best_arg_m4_smote, cv_res_m4_smote, _ = get_best_hyperparameters(grid_search4)

# Declare model
params_m4_smote = best_arg_m4_smote
m4_smote = DecisionTreeClassifier(**params_m4_smote)

# Train model
m4_smote.fit(X4_smote_train, y4_smote_train)

# find the best features
title = "\n20 most important features for DTC Model 4 - original\n"
feature_importance_graph(m4_smote, X4_smote_train, title)

# Resuts tests 
res_m4_smote = test_model(m4_smote, X4_orig_test, y4_orig_test)

# Appending probabilities desertion a dt4
append_prob(m4_smote, X4_orig, dt4, 'prob_m4_smote')


# MODELO ESPECIALIZADO CUARTO AÑO *********************************************

# Dataset original ------------------------------------------------------------

# Hyperparameter tunning 
grid_search4 = dtr_hyperparameters_tuning(X4_orig_train, y4_m4_orig_train)
best_arg_m4_m4, cv_res_m4_m4, _ = get_best_hyperparameters(grid_search4)

# Declare model
params_m4_m4 = best_arg_m4_m4
m4_year4 = DecisionTreeClassifier(**params_m4_m4)

# Train model
m4_year4.fit(X4_orig_train, y4_m4_orig_train)

# find the best features
title = "\n20 most important features for DTC Model 4 year 4- original\n"
feature_importance_graph(m4_year4, X4_orig_train, title)

# Resuts tests 
res_m4_year4 = test_model(m4_year4, X4_orig_test, y4_m4_orig_test)

# Appending probabilities desertion a dt1
append_prob(m4_year4, X4_orig, dt4, 'prob_m4_year4')

# Dataset SMOTE ---------------------------------------------------------------

# Balancing train dataset with SMOTE
print('Original dataset shape %s' % y4_m4_orig_train.value_counts())
sm = SMOTE(random_state=42)
X4_m4_smote_train, y4_m4_smote_train = sm.fit_resample(X4_orig_train, 
                                                    y4_m4_orig_train)
print('Original dataset shape %s' % y4_m4_smote_train.value_counts())

# Hyperparameter tunning 
grid_search4 = dtr_hyperparameters_tuning(X4_m4_smote_train, y4_m4_smote_train)
best_arg_m4_m4_smote, cv_res_m4_m4_smote, _ = \
    get_best_hyperparameters(grid_search4)

# Declare model
params_m4_m4_smote = best_arg_m4_m4_smote
m4_m4_smote = DecisionTreeClassifier(**params_m4_m4_smote)

# Train model
m4_m4_smote.fit(X4_m4_smote_train, y4_m4_smote_train)

# find the best features
title = "\n20 most important features for DTC Model 4 year 4 - smote\n"
feature_importance_graph(m4_m4_smote, X4_m4_smote_train, title)

# Resuts tests 
res_m4_m4_smote = test_model(m4_m4_smote, X4_orig_test, y4_m4_orig_test)

# Appending probabilities desertion a dt4
append_prob(m4_m4_smote, X4_orig, dt4, 'prob_m4_m4_smote')

#-----------------------------------------------------------------------------#
# Saving model with best performance 
# modelo especializado 4to año con dataset balanceado SMOTE

filename = path+"05_models/decision_tree_year4.joblib"
joblib.dump(m4_m4_smote, filename)
