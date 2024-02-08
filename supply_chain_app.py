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

# import warnings
# warnings.filterwarnings("ignore")
from time import sleep
# from selenium import webdriver
# from webdriver_manager.chrome import ChromeDriverManager
# from selenium.webdriver.common.keys import Keys
import re
from bs4 import BeautifulSoup as bs
import requests

df=pd.read_csv("train.csv")
# df.head()

## ajout de titre et de trois pages
st.title("Projet Supply Chain - Satisfaction des clients")
st.sidebar.title("Sommaire")
pages=["Introduction", "Préparation des données", "DataVizualization", "Modélisation", "Machine Learning", "divers"]
page=st.sidebar.radio("Aller vers", pages)

st.sidebar.write("**Réalisé par:**")
st.sidebar.markdown("[Mustapha LAACHIR](https://www.laapha.com)")
st.sidebar.markdown("[Sabrina DIACQUENOD](https://www.laapha.com)")
## page introduction et présentation de projet
if page == pages[0] : 
  st.write("### Introduction")
  st.write("La supply chain représente les étapes d'approvisionnement allant du processus productif à la distribution de la marchandise au client.") 
  st.write("Suite à ces différentes étapes, la satisfaction client est évaluée afin de : ")
  st.write("-	Étudier la qualité de la supply chain (ex : problème de conception, livraison, prix non adapté, durabilité…)")
  st.write("-	Étudier si le produit/service correspond bien à l’attente du marché.")
  st.write("-	Synthétiser les feedback, améliorations des clients.")
  st.write("-	Aider à la réponse ou à la redirection des clients insatisfaits...")
  st.write("Pour de nombreux produits/services, la satisfaction des clients se mesure grâce aux commentaires, et avis laissés par les clients sur des sites dédiées (ex : Trustpilot).")

  st.write("### Objectifs:")
  st.write("L’objectif de ce projet est d’extraire de l’information de commentaires laissés par les clients.") 
  st.write("Dans un premier temps, l’objectif sera de prédire la satisfaction d’un client à partir des commentaires laissés, c'est-à-dire de prédire le nombre d'étoiles ou la note donnée à partir des commentaires.") 
  st.write("Puis dans un second temps, l’objectif sera d’extraire les propos du commentaire (problème de livraison, article défectueux...) afin d’expliquer la note attribuée.")
  st.write("Enfin, l’objectif sera d’extraire de la réponse du fournisseur les propos du commentaire dans le but d’essayer de les prédire uniquement avec le commentaire afin de générer des réponses automatiques.")



## page préparation des données
if page == pages[1] : 
  st.write("### Webscraping")
  # afficher les premières ligne de DF
  # st.dataframe(df.head(10))

  # st.write(df.shape)
  # st.dataframe(df.describe())

  # if st.checkbox("Afficher les NA") :
  #   st.dataframe(df.isna().sum())

  #################################""""
  categorie, pays ,marque, liens_marque, reviews = [],[],[],[],[]
  # #'https://www.trustpilot.com/categories/furniture_store?country=FR'

  ## mettre ici la liste des catégories à parcourir: alimenter cette liste par:
  ## furniture_stor , bank, travel_insurance_company , car_dealer, jewelry_store,

  liste_liens1 = ['bank']

  ## modifier la liste pour importer les bonnes données, 1,2,3 ou 4 ...
  for lien_c in liste_liens1:

    lien = 'https://www.trustpilot.com/categories/'+str(lien_c)+'?country=FR'
    # récupératu du code html de toute la page et le stocker dans une variable: soup
    page = requests.get(lien)
    soup = bs(page.content, "lxml")

    # # Sélectionner la partie de la page qui contient les numéros de page
    # pagination_div = soup.find('nav', class_='pagination_pagination___F1qS')

    # # Extraire les numéros de page en parcourant les éléments de la pagination
    # try:
    #   page_numbers = []
    #   for item in pagination_div.find_all(['span']):
    #       page_numbers.append(item.get_text())
    # except:
    #       page_numbers.append(1)
    # # print(page_numbers)
    # nb_pages = int(page_numbers[-2])
    # print(lien_c,'contient :',nb_pages, 'pages!')

    ### début de la boucle qui parcours les pages d'une marque X
    for X in range(1,2+1):

        sleep(0) # attendre une demi seconde entre chaque page, pas obligé
        lien2 = 'https://www.trustpilot.com/categories/'+str(lien_c)+'?country=FR&page='+str(X)

        # récupératu du code html de toute la page et le stocker dans une variable: soup
        page = requests.get(lien2)
        soup2 = bs(page.content, "lxml")
        soup_marques = soup2.find_all('div', class_ = ("paper_paper__1PY90 paper_outline__lwsUX card_card__lQWDv card_noPadding__D8PcU styles_wrapper__2JOo2"))
        # print(soup_marques)
        # company = soup.find('h1',class_='typography_default__hIMlQ typography_appearance-default__AAY17 title_title__i9V__').text.strip() ## récupérer le nom de la marque

        ## parcourir le code html (soup) pour extraire les informations des balises
        for lien_m in soup_marques:
          # lienss =
          marque.append(lien_m.find('p',class_ ='typography_heading-xs__jSwUz typography_appearance-default__AAY17 styles_displayName__GOhL2').text)
          liens_marque.append(lien_m.find('a',class_ ='link_internal__7XN06 link_wrapper__5ZJEx styles_linkWrapper__UWs5j').get('href'))
          reviews.append(lien_m.find('p',class_ ='typography_body-m__xgxZ_ typography_appearance-subtle__8_H2l styles_ratingText__yQ5S7'))
          # reviews.append(lien_m.find('div',class_ ='styles_rating__pY5Pk'))
          categorie.append(lien_c)
          pays.append("FR")


  data = {
        'marque': marque,
        'liens_marque': liens_marque,
        'categorie': categorie,
        'reviews': reviews,
        'pays': pays
        }

  # création de Dataframe pour y stocker les données
  df_liens = pd.DataFrame(data)

  ###############  data cleaning: ############################
  df_liens['liens_marque'] = df_liens['liens_marque'].str.replace('/review/','')

  ## extraire le nombre de reviews en utilisant une fonction
  def extraire_chiffres(texte):
      pattern = r'\|\</span>([0-9,]+)'
      match = re.search(pattern, str(texte))
      if match:
          chiffres_apres_barre_span = match.group(1)
          return chiffres_apres_barre_span
      elif len(str(texte)) < 8:
          return texte
      else:
          return None
  ## appliquer la fonction à la colonne reviews
  df_liens['reviews'] = df_liens['reviews'].apply(extraire_chiffres)
  ## convertir reviews en nombre
  df_liens['reviews'] = df_liens['reviews'].str.replace(',','')
  # df_liens['reviews'] = df_liens['reviews'].str.replace('None',0)
  df_liens['reviews']=df_liens['reviews'].astype(float)
  ## trier le dataframe
  df_liens = df_liens.sort_values(by=['categorie', 'reviews'], ascending=[True, False])

  ## enregistrer le dataframe traité en csv et excel: ne pas oublier de modififier _liste_liens4 pour ne pas ecraser l'ancien enregistrement
  # df_liens.to_csv('Avis_trustpilot_liste_liens4.csv')
  # df_liens.to_excel('Avis_trustpilot_liste_liens4.xlsx')

  st.dataframe(df_liens.head(10))


## page de la dataviz
if page == pages[2] : 
  st.write("### DataVizualization")
  fig = plt.figure()
  sns.countplot(x = 'Survived', data = df)
  st.pyplot(fig)
  fig = plt.figure()

  sns.countplot(x = 'Sex', data = df)
  plt.title("Répartition du genre des passagers")
  st.pyplot(fig)
  fig = plt.figure()

  sns.countplot(x = 'Pclass', data = df)
  plt.title("Répartition des classes des passagers")
  st.pyplot(fig)

  fig = sns.displot(x = 'Age', data = df)
  plt.title("Distribution de l'âge des passagers")
  st.pyplot(fig)
  
# Afficher un countplot de la variable cible en fonction du genre.
  fig = plt.figure()
  sns.countplot(x = 'Survived', hue='Sex', data = df)
  st.pyplot(fig)

# Afficher un plot de la variable cible en fonction des classes.
  fig = sns.catplot(x='Pclass', y='Survived', data=df, kind='point')
  st.pyplot(fig)

# Afficher un plot de la variable cible en fonction des âges.
  fig = sns.lmplot(x='Age', y='Survived', hue="Pclass", data=df)
  st.pyplot(fig)

# matrice de correlation
  fig, ax = plt.subplots()
  sns.heatmap(df.corr(), ax=ax)
  st.write(fig)

## '''''''''''''''la page de modélisation '''''''''''''''''''''''''''''''''''
if page == pages[3] : 
  st.write("### Modélisation / modèles")

  import joblib
  # (b) Dans le script Python streamlit_app.py, supprimer les variables non-pertinentes (PassengerID, Name, Ticket, Cabin).
  df = df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
  # (c) Dans le script Python, créer une variable y contenant la variable target. Créer un dataframe X_cat contenant les variables explicatives catégorielles et un dataframe X_num contenant les variables explicatives numériques.
  y = df['Survived']
  X_cat = df[['Pclass', 'Sex',  'Embarked']]
  X_num = df[['Age', 'Fare', 'SibSp', 'Parch']]
  
  for col in X_cat.columns:
    X_cat[col] = X_cat[col].fillna(X_cat[col].mode()[0])
  for col in X_num.columns:
    X_num[col] = X_num[col].fillna(X_num[col].median())
  X_cat_scaled = pd.get_dummies(X_cat, columns=X_cat.columns)
  X = pd.concat([X_cat_scaled, X_num], axis = 1)
  # (g) Dans le script Python, séparer les données en un ensemble d'entrainement et un ensemble test en utilisant la fonction train_test_split du package model_selection de Scikit-Learn.
  from sklearn.model_selection import train_test_split
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

  ######################

  # (i) Dans le script Python, créer une fonction appelée prediction qui prend en argument le nom d'un classifieur et renvoie le classifieur entrainé.

  def prediction(classifier):
      if classifier == 'Random Forest':
          clf = joblib.load("model_rf_")
          
      elif classifier == 'SVC':
          clf = joblib.load("model_svc_")
          
      elif classifier == 'Logistic Regression':
          clf = joblib.load("model_lr_")
          
      # clf.fit(X_train, y_train)

      return clf

  # Puisque les classes ne sont pas déséquilibrées, il est intéressant de regarder l'accuracy des prédictions. Copiez le code suivant dans votre script Python. Il crée une fonction qui renvoie au choix l'accuracy ou la matrice de confusion.

  def scores(clf, choice):
      if choice == 'Accuracy':
          return clf.score(X_test, y_test)
      elif choice == 'Confusion matrix':
          return confusion_matrix(y_test, clf.predict(X_test))
      
  # (j) Dans le script Python, utiliser la méthode st.selectbox() pour choisir entre le classifieur RandomForest, le classifieur SVM et le classifieur LogisticRegression. Puis retourner sur l'application web Streamlit pour visualiser la "select box".
  choix = ['Random Forest', 'SVC', 'Logistic Regression']
  option = st.selectbox('Choix du modèle', choix)
  st.write('Le modèle choisi est :', option)
  
  # Il ne reste plus qu'à entrainer le classifieur choisi en utilisant la fonction prediction précédemment définie et à afficher les résultats.
  clf = prediction(option)

  display = st.radio('Que souhaitez-vous montrer ?', ('Accuracy', 'Confusion matrix','Rapport'))

  if display == 'Accuracy':
      st.write(scores(clf, display))
  elif display == 'Confusion matrix':
      st.dataframe(scores(clf, display))
  elif display == 'Rapport':
      st.write('Le Rapport est en cours de construction...')


## fin

if page == pages[4] : 
  st.write("### Machine Learning")
  
if page == pages[5] : 
  st.write("#### Afficher un slider de 1 à 7")
  st.slider('Slider', 1, 7)

  st.write("#### Afficher un slider de Lundi à Dimanche")
  st.select_slider('Choisir un jour de la semaine', options=
  ['Lundi','Mardi','Mercredi','Jeudi','Vendredi','Samedi','Dimanche'])

  st.write("#### Inserer")
  st.text_input('Insérez votre texte')
  st.number_input('Choisissez votre nombre')
  st.date_input('Choisissez une date')             
  st.time_input('Choisissez un horaire')
  st.file_uploader('Importer votre fichier')
  st.code(''' import streamlit ''', language='python')

  st.markdown("Ceci est un [lien hypertexte](https://www.example.com) vers Example.com.")

  # st.image(image, caption='C est une image')
           
  # pip install streamlit-drawable-canvas
  # streamlit-drawable-canvas()