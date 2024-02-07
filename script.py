# C:\datascientest\streamlit_supply_chain> streamlit_projet_supply_chain.py
import streamlit as st
import random
st.write("Pour éviter de perdre du temps à chaque mise à jour de la page, nous pouvons utiliser le décorateur @st.cache_data. Il permet de garder en mémoire une valeur de telle sorte que si nous rafraichissons la page Streamlit avec un (Re-run) nous obtenons toujours la même chose.")
 
@st.cache_data
def generate_random_value(x): 
  return random.uniform(0, x) 
a = generate_random_value(10) 
b = generate_random_value(20) 
st.write(a) 
st.write(b)


####################""
# chemain acces avec terminal: datascientest\Streamlit_supply_chain

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
# from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
from sklearn.metrics import classification_report

df=pd.read_csv("train.csv")
df.head()

## ajout de titre et de trois pages
st.title("Projet de classification binaire Titanic (spyder)")
st.sidebar.title("Sommaire")
pages=["Introduction", "Préparation des données", "DataVizualization", "Modélisation"]
page=st.sidebar.radio("Aller vers", pages)

## page introduction et présentation de projet
if page == pages[0] : 
  st.write("### Introduction")


## page préparation des données
if page == pages[1] : 
  st.write("### Webscraping")


## page de la dataviz
if page == pages[2] : 
  st.write("### DataVizualization")

##
if page == pages[3] : 
  
  import joblib
  joblib.load("model_rf")
  joblib.load("model_svc")
  joblib.load("model_lr")

  st.write("### Modélisation")

  # (b) Dans le script Python streamlit_app.py, supprimer les variables non-pertinentes (PassengerID, Name, Ticket, Cabin).
  df = df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
  # (c) Dans le script Python, créer une variable y contenant la variable target. Créer un dataframe X_cat contenant les variables explicatives catégorielles et un dataframe X_num contenant les variables explicatives numériques.
  y = df['Survived']
  X_cat = df[['Pclass', 'Sex',  'Embarked']]
  X_num = df[['Age', 'Fare', 'SibSp', 'Parch']]
  # (d) Dans le script Python, remplacer les valeurs manquantes des variables catégorielles par le mode et remplacer les valeurs manquantes des variables numériques par la médiane.
  # (e) Dans le script Python, encoder les variables catégorielles.
  # (f) Dans le script Python, concatener les variables explicatives encodées et sans valeurs manquantes pour obtenir un dataframe X clean.
  for col in X_cat.columns:
    X_cat[col] = X_cat[col].fillna(X_cat[col].mode()[0])
  for col in X_num.columns:
    X_num[col] = X_num[col].fillna(X_num[col].median())
  X_cat_scaled = pd.get_dummies(X_cat, columns=X_cat.columns)
  X = pd.concat([X_cat_scaled, X_num], axis = 1)
  # (g) Dans le script Python, séparer les données en un ensemble d'entrainement et un ensemble test en utilisant la fonction train_test_split du package model_selection de Scikit-Learn.
  from sklearn.model_selection import train_test_split
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
  
  # (h) Dans le script Python, standardiser les valeurs numériques en utilisant la fonction StandardScaler du package Preprocessing de Scikit-Learn.
  from sklearn.preprocessing import StandardScaler
  scaler = StandardScaler()
  X_train[X_num.columns] = scaler.fit_transform(X_train[X_num.columns])
  X_test[X_num.columns] = scaler.transform(X_test[X_num.columns])

  ########### pour enregistrer les model avec joblib ################
  import joblib
  clf_rf = RandomForestClassifier()
  joblib.dump(clf_rf, "model_rf")

  clf_svc = SVC()
  joblib.dump(clf_svc, "model_svc")

  clf_lr = LogisticRegression()
  joblib.dump(clf_lr, "model_lr")

  ######################

## fin
