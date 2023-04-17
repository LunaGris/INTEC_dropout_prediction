# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 11:21:58 2023

@author: DMatos

Script for processing and predicting drop out risk for active students
"""

#*****************************************************************************#
#***************************Setting up environment****************************#
#*****************************************************************************#

#*LIBRARIES*#
import pandas as pd


"""
#*WORKING DIRECTORY*#
import os
cwd = os.getcwd() #for getting current working directory
os.chdir('/tmp') #for changing working directory 
"""

#*DATA*# 
#path = "C:/Users/DMatos/OneDrive - INTEC/proyecto_final/03_data/" #cambie PC
path = "C:/Users/deya1/OneDrive - INTEC/proyecto_final/03_data/"
dt = pd.read_csv(path + "02_prep/dt_std_active.csv")