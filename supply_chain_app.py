# chemain acces avec terminal: cd datascientest\Supply_chain_juin23\Streamlit
# chemain acces avec terminal: cd C:\Users\laach\OneDrive\Documents\GitHub\Supply_chain_juin23\Streamlit
# st.set_page_config(layout="wide", page_title="Image Background Remover")
# import des modèles

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
import joblib
from time import sleep
import datetime
import re
from bs4 import BeautifulSoup as bs
import requests
from wordcloud import WordCloud


# Ajouter du style CSS personnalisé
st.markdown(
    """
    <style>
    body {
        font-family: 'Arial', sans-serif;
        background-color: #BA4A00;
        /* background-color: #BA4A00;*/
        color: #333;
    }
    h1 {
        /*color: #BA4A00;*/
        /*color: #5DADE2;*/
        color: #566573;        
    }
    h2 {
        color: #BA4A00;
        /*color: #5DADE2;*/
                
    }
    
    </style>
    """,
    unsafe_allow_html=True
)



df_liste_liens = pd.read_excel("Datas/liste_finale_à_scraper.xlsx", index_col=0)
# df_clean = pd.read_excel("Final_data_scraped_traité_traduit_ok_clean.xlsx", index_col=0)
# df.head()

df_clean_2 = joblib.load("models/data_clean_lib")
df_clean_2 = df_clean_2.reset_index(drop = False)

st.image("médias/bannière_smily.png", use_column_width=True)

# ajout de titre et résumé
st.title("Projet Supply Chain - Satisfaction des clients:::")
st.write(
    ":dog: Scraper, traiter et analyser les avis clients. le code source est disponible [ici](https://github.com/LAAPHA/Streamlit_supply_chain) sur GitHub. Special thanks to Datascientest :grin:"
)
st.markdown("***")


# ajout de sommaire et les pages
st.sidebar.title("Sommaire")
# pages=["Introduction", "Préparation des données", "Data Visualization", "Modélisation", "Machine Learning", "divers"]
pages=["Introduction", "Préparation des données", "Data Visualization", "Modélisation", "Clustering"]

page = st.sidebar.radio("Aller vers", pages)


## page introduction et présentation de projet
if page == pages[0] : 
  # Image
  st.image("médias/avis_datascientest.png",  caption='Avis pour Datascientest en mars 2024. source: Trustpilot', use_column_width=True)
  
  st.markdown("<h1 style='text-align: center;'>Peut-on utiliser les avis clients comme facteur prédictif de la satisfaction des clients? </h1>", unsafe_allow_html=True)
  st.markdown("---")

#   st.write("### Introduction")
  st.markdown("<h2 style='text-align: left;'> Introduction:</h2>", unsafe_allow_html=True)
  
  st.markdown("<p style='text-align: left;'>___La supply chain représente les étapes d'approvisionnement allant du processus productif à la distribution de la marchandise"
              " au client.<br>  Suite à ces différentes étapes, la satisfaction client est évaluée afin de : <br> -	Étudier la qualité de la supply chain (ex : problème"
              " de conception, livraison, prix non adapté, durabilité…).<br> -	Étudier si le produit/service correspond bien à l’attente du marché.<br>"
              "  -	Synthétiser les feedback, améliorations des clients.<br>  -	Aider à la réponse ou à la redirection des clients insatisfaits...<br>"
              "  Pour de nombreux produits/services, la satisfaction des clients se mesure grâce aux commentaires, et avis laissés par les clients sur des sites dédiées "
              "(ex : Trustpilot).</p>", unsafe_allow_html=True)

  st.markdown("<h2 style='text-align: left;'> Objectifs:</h2>", unsafe_allow_html=True)

  st.markdown("<p style='text-align: left;'>___L’objectif de ce projet est d’extraire de l’information de commentaires laissés par les clients.<br> Dans un premier temps,"
              " l’objectif sera de prédire la satisfaction d’un client à partir des commentaires laissés, c'est-à-dire de prédire le nombre d'étoiles ou la note donnée à partir"
              " des commentaires. <br>Puis dans un second temps, l’objectif sera d’extraire les propos du commentaire (problème de livraison, article défectueux...)"
              " afin d’expliquer la note attribuée. <br> Enfin, l’objectif sera d’extraire de la réponse du fournisseur les propos du commentaire dans le but d’essayer "
              "de les prédire uniquement avec le commentaire afin de générer des réponses automatiques. </p>", unsafe_allow_html=True)
              
  

  # about this application

  with st.expander('About this app'):
    st.markdown('**What can this app do?**')
    st.info('This app allow users to build a machine learning (ML) model in an end-to-end workflow.'
            ' Particularly, this encompasses data upload, data pre-processing, ML model building and post-model analysis.')

    st.markdown('**How to use the app?**')
    st.warning('To engage with the app, go to the sidebar and 1. Select a data set and 2. Adjust the model parameters by adjusting'
              ' the various slider widgets. As a result, this would initiate the ML model building process, display the model results'
              ' as well as allowing users to download the generated models and accompanying data.')

    st.markdown('**Under the hood**')
    st.markdown('Data sets:')
    st.code('''- Drug solubility data set
    ''', language='markdown')
    
    st.markdown('Libraries used:')
    st.code('''- Pandas for data wrangling
    - Scikit-learn for building a machine learning model
    - Altair for chart creation
    - Streamlit for user interface
    ''', language='markdown')


## page préparation des données
if page == pages[1] : 
  
  st.markdown("<h1 style='text-align: center;'> Webscraping: Préparation et ciblage des données à scraper </h1>", unsafe_allow_html=True)

#   st.write("### Première étape: Préparation de la liste des liens à scraper:")
  st.markdown("<h2 style='text-align: left;'>Première étape: Préparation de la liste des liens à scraper:</h2>", unsafe_allow_html=True)

  st.write("------------------------------------------------------------------------")
 
  ###################### debut webscraping###########""""
  categorie, pays ,marque, liens_marque, reviews = [],[],[],[],[]
  # #'https://www.trustpilot.com/categories/furniture_store?country=FR'

  ## mettre ici la liste des catégories à parcourir: alimenter cette liste par:
  
  liste_liens1 = ['bank','mortgage_broker','travel_insurance_company','insurance_agency']
  liste_liens2 = ['All','bank','mortgage_broker','travel_insurance_company','insurance_agency']

#   st.write("les catégories à scraper:", liste_liens1)
  
  st.markdown("<p style='text-align: center;'>Le site Trustpilot se présente de la façon suivante. Il regroupe les avis client par catégorie : Bank, Travel insurance company, Car dealer… Il y a 24 catégories et chaque catégorie contient plusieurs marques. </p>", unsafe_allow_html=True)
  st.image("médias/catégories.png",  caption='Les différentes catégories du Trustpilot', use_column_width=True)

  st.markdown("<p style='text-align: center;'>Pour chaque catégorie, il faut scraper les liens des marques qui lui sont rattachées. Pour accéder à chaque catégorie, il faut remplacer le nom de la catégorie par une variable X puis parcourir la liste des catégories avec une boucle For.<br> <b>'https://www.trustpilot.com/categories/bank  ===> 'https://www.trustpilot.com/categories/X <b></p>", unsafe_allow_html=True)
  st.image("médias/schéma_site.png",  caption='La structure du site', use_column_width=True)
  
  st.write("------------------------------------------------------------------------")

  st.markdown("<p style='text-align: center;'>!!Ce formulaire est adapté uniquement au site truspilot. Des modifications mineures sont nécessaires pour le généraliser!! <br> </p>", unsafe_allow_html=True)

  st.markdown('<span style="color:red"> Nous faisons ce scraping pour des fins éducatives dans le cadre de notre formation. Le webscraping de masse est interdit, veuillez respecter les conditions du site internet </span>',unsafe_allow_html=True )
  # Champ de saisie pour l'URL
  url = st.text_input("Entrez la racine de l'URL à scraper: ","https://www.trustpilot.com/categories", key = "URL")
  # lien_c = st.text_input("Entrez la catégorie à scraper :")
  lien_c = st.selectbox('Choix de la catégorie:', liste_liens2)
  
  if lien_c == 'All':
      liste_f = liste_liens1
  else:
      liste_f = [lien_c]

  # Bouton pour lancer le scraping
  if st.button("Scraper les liens des marques"):
      
      if url:
          
          for lien_c in liste_f:
                       
            lien = str(url) + '/' + str(lien_c) +'?country=FR'
            
            # récupératu du code html de toute la page et le stocker dans une variable: soup
            page = requests.get(lien, verify = False)
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
                lien2 = str(url)+'/'+str(lien_c)+'?country=FR&page='+str(X)
                
                # récupératu du code html de toute la page et le stocker dans une variable: soup
                page = requests.get(lien2, verify = False)
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
          # df_liens.to_csv('résults/Avis_trustpilot_liste_liens4.csv')
          # df_liens.to_excel('résults/Avis_trustpilot_liste_liens4.xlsx')

          st.dataframe(df_liens.head(10))
          st.write("La taille de df_liste_liens: ",df_liens.shape)
        #   st.write(df_liens['categorie'].value_counts())

          f,ax = plt.subplots(figsize=(6, 4))
          sns.countplot (y = 'categorie', data = df_liens)
        # ax.legend(ncol=2, loc="lower right", frameon=True)
          ax.set( ylabel="", xlabel="Nombre de campanies par catégories (secteurs)")
          sns.despine(left=True, bottom=True)
          st.pyplot(f)
          ############### fin webscraping #################################################

      else:
          st.warning("Veuillez saisir une URL valide.")

  ##------------------------------------------------------------------------------------------        
  ### debut de webscraping: de tous les liens-------------------------------------------------
  
#   st.write("### Deuxième étape: Scraper les données:")
  st.markdown("<h2 style='text-align: left;'>Deuxième étape: Scraper les données:</h2>", unsafe_allow_html=True)

  st.write("Vu le temps que cela prendra, nous allons scraper uniquement un échantillon des liens. On va scraper la première page de la catégorie bank")        
  
  if st.button("Scraper tous les avis"):
    st.write("scraper les avis clients:")
    st.dataframe(df_liste_liens.head(10))
    # Obtenez la date et l'heure actuelles
    
    date_actuelle = datetime.datetime.now()
    # Affichez la date uniquement au format "Année-Mois-Jour"
    st.write("Date du jour (format court) :", date_actuelle.strftime("%d-%m-%Y"))
    
    ## le dataframe de travail, choix des catégories à scraper: pour scraper en plusieurs morceaux
    
    df_liens_filtré = df_liste_liens.loc[(df_liste_liens['categorie']=='bank') ]
    # df_liens_filtré = df_liens.loc[(df_liens['categorie']=='bank') ]
    # df_liens_filtré = concatenated_df

    # liste_liens = ['qonto.com' ,'anyti.me','chanel.com']
    liste_cat = df_liens_filtré ['categorie'].unique()
    st.write ('liste des catégories à scraper:\n', liste_cat)
    

    nombre_fichier = 0

    for lien_cat in liste_cat:
        ## parcourir la liste des liens pour plusieurs catégories
        Data = {}
        # création des  listes vides
        tout , noms, date_commentaire, date_experience, notes, titre_com, companies, reponses = [],[],[],[],[],[],[],[]
        commentaire, verified ,test, site,nombre_pages , date_scrap, date_reponse, date_rep,categorie_bis = [],[],[],[],[],[],[],[],[]

        df_marque = df_liens_filtré.loc[df_liens_filtré ['categorie'] == lien_cat]

        for lien_c in df_marque['liens_marque']:

            lien = 'https://www.trustpilot.com/review/'+str(lien_c)+'?page=1'

            # récupératu du code html de toute la page et le stocker dans une variable: soup
            try:
                page = requests.get(lien, verify = False)
                soup = bs(page.content, "lxml")
            except:
                st.write(f"Une exception s'est produite pour {lien_c}: {e}")
                # continue

            # Sélectionner la partie de la page qui contient les numéros de page
            pagination_div = soup.find('div', class_='styles_pagination__6VmQv')
            # Extraire les numéros de page en parcourant les éléments de la pagination
            page_numbers = []

            try:
                for item in pagination_div.find_all(['span']):
                    page_numbers.append(item.get_text())

                # print(page_numbers)
                nb_pages = int(page_numbers[-2])
                # st.write(lien_c, 'contient : ',nb_pages, ' pages!')
            except:
                nb_pages = 1

            ### début de la boucle qui parcours les pages d'une marque X
            # for X in range(1,nb_pages+1):
            
            for X in range(1,2):

                # sleep(0) # attendre une demi seconde entre chaque page, pas obligé

                lien = 'https://www.trustpilot.com/review/'+str(lien_c)+'?page='+str(X)
                # récupératu du code html de toute la page et le stocker dans une variable: soup
                page = requests.get(lien, verify = False)
                soup = bs(page.content, "lxml")
                # print(soup.prettify())
                avis_clients = soup.find_all('div', attrs = {'class': "styles_cardWrapper__LcCPA styles_show__HUXRb styles_reviewCard__9HxJJ"})

                try:
                    company = soup.find('h1',class_='typography_default__hIMlQ typography_appearance-default__AAY17 title_title__i9V__').text.strip() ## récupérer le nom de la marque
                except:
                    company = None
                ## parcourir le code html (soup) pour extraire les informations des balises
                for avis in avis_clients:

                    # tout.append(avis.find('div',class_='styles_reviewContent__0Q2Tg').text.strip())
                    try:
                        noms.append(avis.find('span',class_='typography_heading-xxs__QKBS8 typography_appearance-default__AAY17').text.strip())
                    except:
                        noms.append(None)
                    try:
                        titre_com.append(avis.find('h2',class_='typography_heading-s__f7029 typography_appearance-default__AAY17').text.strip())
                    except:
                        titre_com.append(None)
                    try:
                        commentaire.append(avis.find('p').text.strip())
                    except:
                        commentaire.append(None)
                    try:
                        reponses.append(avis.find('p',class_='typography_body-m__xgxZ_ typography_appearance-default__AAY17 styles_message__shHhX'))
                    except:
                        reponses.append(None)
                    # try:
                    # notes.append(avis.find('img')['alt'])
                    # except:
                    #   notes.append(None)
                    try:
                        notes.append(avis.find('div',class_='star-rating_starRating__4rrcf star-rating_medium__iN6Ty'))
                    except:
                        notes.append(None)
                    try:
                        date_experience.append(avis.find('p',class_='typography_body-m__xgxZ_ typography_appearance-default__AAY17').text.strip())
                    except:
                        date_experience.append(None)
                    try:
                        date_commentaire.append(avis.find('div',class_='styles_reviewHeader__iU9Px').text.strip())
                    except:
                        date_commentaire.append(None)
                    # try:
                    # date_reponse.append(avis.find('div',class_='styles_content__Hl2Mi'))
                    # except:
                    #   noms.append(None)
                    try:
                        companies.append(company)
                    except:
                        companies.append(None)
                    try:
                        site.append(lien)
                    except:
                        site.append(None)
                    try:
                        nombre_pages.append(nb_pages)
                    except:
                        nombre_pages.append(None)
                    try:
                        categorie_bis.append(lien_cat)
                    except:
                        categorie_bis.append(None)

                    date_scrap.append(date_actuelle.strftime("%d-%m-%Y"))

            nombre_fichier+=1
            st.write('Nous avons scrapé ' ,nb_pages, ' pages du site: ', lien_cat , '/',lien_c, ' *** N°:***',nombre_fichier)

        # création d'un dictionnaire avec les listes crées précédement
        data = {
                'categorie_bis': categorie_bis,
                'companies': companies,
                'noms': noms,
                'titre_com': titre_com,
                'commentaire': commentaire,
                'reponses': reponses,
                'notes': notes,
                'date_experience': date_experience,
                'date_commentaire': date_commentaire,
                # 'date_reponse': date_reponse,
                'site': site,
                'nombre_pages': nombre_pages,
                'date_scrap': date_scrap
                }

        # création de Dataframe pour y stocker les données
        df = pd.DataFrame(data)
        # enregistrer le dataframe dans un fichier .csv

        # df.to_excel('Avis_trustpilot_supply_chain_brut_'+str(lien_cat)+'.xlsx')
        # df.to_excel('Avis_trustpilot_supply_chain_brut_total.xlsx')

        st.write('!!!!!!!!!!!!!!!!!La categorie !!' , lien_cat, '!! est scrapée et enregistrée en Excel!!!!!!!!!!!!!!')
   
    st.write("------------------------------------------------------------------------")
    st.write('#### Résultats: données brutes scrapées:')
    
    st.write("Voici le Dataframe des données brutes scrapée (données non traitées). \nD'après ce que nous voyons ci-dessus, Les données scrapées nécessitent un traitement supplémentaire avec text mining. Nous allons aussi procéder à la création de nouvelles features engineering.")
    st.dataframe(df.head(10))
    st.write(df['companies'].value_counts())
    st.write("La taille du df brute: ",df.shape)

  # st.dataframe(df_liste_liens.describe())
  # if st.checkbox("Afficher les NA") :
  #   st.dataframe(df_liste_liens.isna().sum())

#   st.write("### Troixième étape: Data Cleaning & Feature engineering ")
  st.markdown("<h2 style='text-align: left;'>Troixième étape: Data Cleaning & Feature engineering:</h2>", unsafe_allow_html=True)

  st.write("Cette étape de prétraitement des données est essentielle afin que les données soient prêtes pour l'analyse. Elle permet de réduire le bruit, de simplifier l’analyse et l'interprétation des résultats et de faciliter la création de modèles d'apprentissage automatique pour la prédiction de la satisfaction client. Une fois que les données sont traitées, il est possible de passer à l'exploration, à l'analyse et à la visualisation pour en tirer des informations précieuses.")

  st.write("#### a)   Text mining")
  st.image("médias/ex_commentaire_brut.png",  caption='Exemple avis client non traité',  use_column_width=True, width=200 )

  st.write("#### b)	Traduction des commentaires")
  st.write("Les commentaires étaient dans des langues différentes. En effet, les clients provenaient de différentes régions du monde et s’exprimaient généralement dans leur langue natale.Une étape de traduction des commentaires était donc nécessaire. Nous avons utilisé la fonction « langdetect » pour détecter la langue de commentaire et traduire en anglais les commentaires écrits dans une autre langue.")

  st.write("#### c)	Création de nouvelles variables")
  st.write("Une étape de feature engineering aboutissant à la création de nouvelles variables a été effectuée.")

  st.write("#### d)	Encodage des variables catégorielles")
  st.write("L’objectif de cette étape est de convertir certaines variables numériques en variables catégorielles afin qu’elles soient utilisées plus facilement dans les modèles de machine Learning. C’est pourquoi nous avons décidés de transformer la variable cible « notes » en catégorie. Nous avons donc créé la variable « notes_bis » où nous avons regroupé les notes deux catégories : ")
  st.image("médias/cat_notes.png",  caption='Regroupement des notes',  use_column_width=True, width=200 )


###################### page de la DataViz ###############################

if page == pages[2] : 
  
  # import joblib
  # df_clean_2 = joblib.load("data_clean_")

  # Supprission des colonnes inutiles pour dataviz
  colonnes_à_supprimer_v = [ 'companies', 'noms', 'titre_com', 'commentaire', 'verif_reponses',
       'reponses',  'date_experience', 'date_commentaire', 'site', 'nombre_pages', 'date_scrap', 'année_experience' ,
       'mois_experience', 'jour_experience', 'année_commentaire','mois_commentaire', 'jour_commentaire', 'leadtime_com_exp','caractères_spé',
       'commentaire_text','commentaire_en', 'verif_traduction', 'commentaire_en_bis','cat_nombre_caractères','cat_nombre_maj']
      #  , 'nombre_point_intero','nombre_point_exclam','categorie_bis','langue']

  # supprimer les colonnes inutiles et création de df pour dataviz
  df_v = df_clean_2.drop(columns = colonnes_à_supprimer_v)

#   st.write("### Data Visualization")
  st.markdown("<h2 style='text-align: left;'>Description de DataFrame:</h2>", unsafe_allow_html=True)
  
  st.dataframe(df_v.head(10))

# Informations sur le DataFrame dans Streamlit
  st.write("Informations sur le DataFrame :")
  st.write("Nombre de lignes :", df_v.shape[0])
  st.write("Nombre de colonnes :", df_v.shape[1])
  st.write("\n")
  st.write("Types de données de chaque colonne :")
  st.code(df_v.dtypes)
#   st.write("\n")
#   st.write("Valeurs non nulles par colonne :")
#   st.code(df_v.notnull().sum())
  


  st.markdown("<h2 style='text-align: left;'>Data Visualization:</h2>", unsafe_allow_html=True)

# graphe : répartition des commentaires par catégories
#   st.write("------------------------------------------------------------------------")
  st.write("Sur ce graphe, nous avons représenté les proportions de chaque catégorie présente dans notre jeu de données. Les catégories les plus représentés dans notre jeu de données sont : activewear_store, cosmetics_store, clothing_store, appliance_store, travel_agency.")

  count_df = df_v['categorie_bis'].value_counts().reset_index()
  count_df.columns = ['categorie_bis', 'count']
    # Créer le graphique Plotly avec une couleur pour chaque catégorie
  fig = px.bar(count_df, y='categorie_bis', x='count', color='categorie_bis', 
             orientation='h', labels={'categorie_bis': "Catégorie", 'count': "Nombre d'avis"},
             title="Nombre d'avis par catégories (secteurs)")
    # Modifier la taille de la figure
  fig.update_layout(width=800, height=600)
  st.plotly_chart(fig)
  
  col1, col2 = st.columns(2)

#   fig, ax = plt.subplots(figsize=(8,8))
   
# graphe répartition des commentaires par notes
#   plt.subplot(121)
  st.write("------------------------------------------------------------------------")
  st.write("Pour mieux comprendre la notation des clients, nous avons représentés ici la distribution des notes sans les catégories. La majorité des notes attribuées sont bonnes. Mais il y a quand même quelques mauvaises notes. De plus, il y a très peu de notes intermédiaires (2,3). Les clients donnent des notes extrêmes (1 ou 5). Donc les clients sont soit très satisfait (5 = la majorité) soit très insatisfait (1).")
 
  count_df = df_v['notes'].value_counts().reset_index()
  count_df.columns = ['notes', 'count']
    # Créer le graphique Plotly
  fig = px.bar(count_df, x='notes', y='count',  color='notes',
             labels={'notes': "Note", 'count': "Nombre d'occurrences"},
             title="Répartition des notes")
  st.plotly_chart(fig)




  st.write("------------------------------------------------------------------------")
  st.write("Nous observons comme que plus la note est faible et plus le nombre de caractère augmente. Donc pour des avis négatifs, nous observerons en général des commentaires avec un nombre de caractères élevés, ce qui permet d'expliquer le contexte du problème au vendeur afin de trouver une solution. Le nombre de caractère du commentaire semble exercer une influence sur la note du client.")
  
  ## graphes pour impact de nombre de caractères et majiscules sur la note
  fig, ax = plt.subplots(figsize=(10,4))

  plt.subplot(121)
#   fig, ax = plt.subplots(figsize=(4,4)) 
  notescar = df_v[['nombre_caractères','notes']].groupby('notes').mean().sort_values(by='nombre_caractères', ascending=False)
  notescar.reset_index(0, inplace=True)
  sns.barplot(x= notescar['notes'], y= notescar['nombre_caractères'], palette="Blues_r")
  plt.xlabel('\nNotes', fontsize=10, color='#2980b9')
  plt.ylabel('Nombre de caractère\n', fontsize=10, color='#2980b9')
  plt.title("Moyenne du nombre de caractères en fonction des notes\n", fontsize=12, color='#3742fa')
  plt.xticks(rotation= 90)
  plt.tight_layout()
#   st.write(fig)

  plt.subplot(122)
#   fig, ax = plt.subplots(figsize=(4,4))
  notescar = df_v[['nombre_maj','notes']].groupby('notes').mean().sort_values(by='nombre_maj', ascending=False)
  notescar.reset_index(0, inplace=True)
  sns.barplot(x= notescar['notes'], y= notescar['nombre_maj'], palette="Blues_r")
  plt.xlabel('\nNotes', fontsize=10, color='#2980b9')
  plt.ylabel('Nombre de caractères majiscules\n', fontsize=10, color='#2980b9')
  plt.title("Moyenne de caractères majiscules en fonction des notes\n", fontsize=12, color='#3742fa')
  plt.xticks(rotation= 90)
  plt.tight_layout()
#   st.write(fig)
  
  st.pyplot(fig)

  ## graphes pour les emojis positifs et négatifs

  st.write("\n")
  st.write("Nous avons représenté ici le nombre d’émojis positifs selon la note (en catégorie). Nous avons observé qu’il y a plus de smileys positifs lorsque la note est positive que lorsqu’elle est négative.")
  
  fig, ax = plt.subplots(figsize=(10,4))
  
  plt.subplot(121)
  #   fig, ax = plt.subplots(figsize=(4,4))
  notescar = df_v[['emojis_positifs_count','notes']].groupby('notes').mean().sort_values(by='emojis_positifs_count', ascending=False)
  notescar.reset_index(0, inplace=True)
  sns.barplot(x= notescar['notes'], y= notescar['emojis_positifs_count'], palette="Blues_r")
  plt.xlabel('\nNotes', fontsize=10, color='#2980b9')
  plt.ylabel('Nombre emojis positifs\n', fontsize=10, color='#2980b9')
  plt.title("Moyenne du Nombre emojis positifs en fonction des notes\n", fontsize=10, color='#3742fa')
  plt.xticks(rotation= 90)
  plt.tight_layout()
#   st.write(fig)

  plt.subplot(122)
#   fig, ax = plt.subplots(figsize=(4,4))
  notescar = df_v[['emojis_negatifs_count','notes']].groupby('notes').mean().sort_values(by='emojis_negatifs_count', ascending=False)
  notescar.reset_index(0, inplace=True)
  sns.barplot(x= notescar['notes'], y= notescar['emojis_negatifs_count'], palette="Blues_r")
  plt.xlabel('\nNotes', fontsize=10, color='#2980b9')
  plt.ylabel('Nombre emojis négatifs\n', fontsize=10, color='#2980b9')
  plt.title("Moyenne du Nombre emojis négatifs en fonction des notes\n", fontsize=10, color='#3742fa')
  plt.xticks(rotation= 90)
  plt.tight_layout()
#   st.write(fig)
    
  st.pyplot(fig)


  
# graphe répartition des commentaires par notes_bis
#   plt.subplot(122)
  st.write("------------------------------------------------------------------------")
  st.write("Nous avons représenté la distribution des notes par catégories (0 : mauvaises notes= 1, 2, 3 ; 1 : bonnes notes = 4, 5). Il y a 5 fois plus de notes positives que de notes négatives.")

  count_df = df_v['notes_bis'].value_counts().reset_index()
  count_df.columns = ['notes_bis', 'count']
    # Créer le graphique Plotly
  fig = px.bar(count_df, x='notes_bis', y='count',  color='notes_bis',
             labels={'notes_bis': "Note", 'count': "Nombre d'occurrences"},
             title="Regroupement des notes")
  
  st.plotly_chart(fig)
#   st.pyplot(fig)
  


# graphe  des langues
  st.write("------------------------------------------------------------------------")
  st.write("Nous nous sommes rendu compte que la grande majorité des commentaires étaient écrits en anglais. Nous avons donc choisi de tout traduire en anglais. Pour effectuer cette tâche, nous avons utilisé la bibliothèque suivante : googletrans. \nCe graphe représente la répartition des commentaires en fonction de la langue utilisée.")

  count_df = df_v['langue'].value_counts().reset_index()
  count_df.columns = ['langue', 'count']
    # Créer le graphique Plotly avec une couleur pour chaque langue
  fig =  px.bar(count_df, x='langue', y='count', color='langue',
             labels={'langue': "Langue", 'count': "Nombre de commentaires"},
             title="Répartition des commentaires par langue")
    # Afficher le graphique dans Streamlit
  st.plotly_chart(fig)


# matrice de correlation
  st.write("------------------------------------------------------------------------")
  st.write("Nous observons des coefficients de corrélation élevés entre les variables que nous avions jugées précédemment pertinentes et la variable cible. Ainsi, il serait intéressant d’utiliser ces variables dans le modèle pour essayer d’expliquer et de prédire la variable cible. ")

  fig, ax = plt.subplots(figsize=(8,8))
  sns.heatmap(df_v.corr(), annot=True, ax=ax ,cmap='coolwarm')
  st.write(fig)
    
    


## '''''''''''''''la page de modélisation '''''''''''''''''''''''''''''''''''
  
if page == pages[3] : 
  
  st.markdown("<h2 style='text-align: left;'>Modélisation des modèles:</h2>", unsafe_allow_html=True)

  #################""
  ## Colonnes à supprimer
  colonnes_à_supprimer = ['categorie_bis','verified', 'nombre_caractères', 'nombre_caractères', 'nombre_maj', 'nombre_car_spé', 'emojis_positifs_count',
       'emojis_negatifs_count',  'nombre_point_intero', 'nombre_point_exclam', 'companies', 'noms', 'titre_com', 'commentaire', 'verif_reponses',
       'reponses',  'date_experience', 'date_commentaire', 'site', 'nombre_pages', 'date_scrap',  'année_experience', 'langue',
       'mois_experience', 'jour_experience', 'année_commentaire','mois_commentaire', 'jour_commentaire', 'leadtime_com_exp','caractères_spé',
       'commentaire_text','commentaire_en', 'verif_traduction', 'commentaire_en_bis','cat_nombre_caractères','cat_nombre_maj','notes','sentiment_commentaire','commentaire_clean']

# supprimer les colonnes inutiles
  df2 = df_clean_2.drop(columns = colonnes_à_supprimer)
  df2 = df2.drop_duplicates()
  df2 = df2.dropna(subset = ['commentaire_clean_pos_tag'])

  ## preparation des données pour modeles
  # on commence par transformer notre variable à prédire en variable binaire
  encode_y = LabelEncoder()
  x = df2["commentaire_clean_pos_tag"] ## ajouter POS Tagging
  y = encode_y.fit_transform(df2["notes_bis"])
# on sépare en apprentissage/validation
  x_train, x_test, y_train, y_test = train_test_split(x , y ,test_size = 0.2 , random_state=42)

# on transforme en matrice creuse d’occurrence des mots (on transforme x_train et on applique à x_test la transformation)
# trans_vect = CountVectorizer()
  trans_vect = TfidfVectorizer()
  x_train_trans = trans_vect.fit_transform(x_train)
  x_test_trans  = trans_vect.transform(x_test)
    
  ######################

  # (i) Dans le script Python, créer une fonction appelée prediction qui prend en argument le nom d'un classifieur et renvoie le classifieur entrainé.

  def prediction(classifier):
      if classifier == 'Naive Bayes':
          clf = joblib.load("models/modele_bayes_lib")
                
      elif classifier == 'Gardient boosting':
          clf = joblib.load("models/modele_gb_lib")

      elif classifier == 'SVC':
          clf = joblib.load("models/modele_svm_lib")

      elif classifier == 'KNN':
          clf = joblib.load("models/modele_knn_lib")

      # clf.fit(X_train, y_train)

      return clf

  # Puisque les classes ne sont pas déséquilibrées, il est intéressant de regarder l'accuracy des prédictions. Copiez le code suivant dans votre script Python. Il crée une fonction qui renvoie au choix l'accuracy ou la matrice de confusion.
  
  def scores(clf, choice):
      if choice == 'Accuracy':
          return clf.score(x_test_trans,y_test)
      elif choice == 'Confusion matrix':
          return confusion_matrix(y_test, clf.predict(x_test_trans))
  
  

  # (j) Dans le script Python, utiliser la méthode st.selectbox() pour choisir entre le classifieur RandomForest, le classifieur SVM et le classifieur LogisticRegression. Puis retourner sur l'application web Streamlit pour visualiser la "select box".
  choix = ['Naive Bayes', 'Gardient boosting','SVC','KNN']
  option = st.selectbox('Choix du modèle', choix)
  st.write('Le modèle choisi est :', option)
  
  # Il ne reste plus qu'à entrainer le classifieur choisi en utilisant la fonction prediction précédemment définie et à afficher les résultats.
  clf = prediction(option)

  # y_pred = clf.predict(x_test_trans) # Calculate predictions
  # st.write('Le modèle choisi est :', accuracy_score(y_test, y_pred))

  
  
  display = st.radio('Que souhaitez-vous montrer ?', ('Accuracy', 'Confusion matrix','Rapport'))

  if display == 'Accuracy':
      # clf
      st.write(scores(clf, display))
      
  elif display == 'Confusion matrix':
      st.dataframe(scores(clf, display))

  elif display == 'Rapport':
      st.write('Le Rapport est en cours de construction...')

  st.write('présentation en colonnes')
  col1, col2 = st.columns(2)
  #   col1.st.scores(clf, display)
  #   col2.st.dataframe(scores(clf, display))
  


  # Ajouter des onglets
  tabs = ["Page 1", "Page 2", "Page 3"]
  selected_tab = st.sidebar.radio("Sélectionnez une page", tabs)

  # Contenu des onglets
  if selected_tab == "Page 1":
    st.write("Contenu de la page 1")
  elif selected_tab == "Page 2":
    st.write("Contenu de la page 2")
  elif selected_tab == "Page 3":
    st.write("Contenu de la page 3")

  # Sélection de l'onglet
  liste_tab = []

  tab1, tab2, tab3 = st.tabs(["rapport 1", "rapport 2", "rapport 3"])

  with tab1:
   st.header("rapport 1")
   st.image("https://static.streamlit.io/examples/cat.jpg", width=200)

  with tab2:
   st.header("rapport 2")
   st.image("https://static.streamlit.io/examples/dog.jpg", width=200)

  with tab3:
   st.header("rapport 3")
   st.image("https://static.streamlit.io/examples/owl.jpg", width=200)

## fin modélisation
    



### ####################### Le clustering ###################################################
if page == pages[4] :
  
  colonnes_à_supprimer = ['verified', 'nombre_caractères', 'nombre_caractères', 'nombre_maj', 'nombre_car_spé', 'emojis_positifs_count',
       'emojis_negatifs_count',  'nombre_point_intero', 'nombre_point_exclam', 'companies', 'noms', 'titre_com', 'commentaire', 'verif_reponses',
       'reponses',  'date_experience', 'date_commentaire', 'site', 'nombre_pages', 'date_scrap',  'année_experience', 'langue',
       'mois_experience', 'jour_experience', 'année_commentaire','mois_commentaire', 'jour_commentaire', 'leadtime_com_exp','caractères_spé',
       'commentaire_text','commentaire_en', 'verif_traduction', 'commentaire_en_bis','cat_nombre_caractères','cat_nombre_maj','notes','commentaire_clean_pos_tag','sentiment_commentaire']

   ## supprimer les colonnes inutiles
  df3 = df_clean_2.drop(columns=colonnes_à_supprimer)
  #Affiche le nombre de doublons
  st.write("Avant suppression duplicates", df3.shape)

  nombre_doublon = df3.duplicated().sum()
  st.write("Nombre de doublons :", nombre_doublon)

  #Supprime les doublons
  df3 = df3.drop_duplicates()
  df3 = df3.dropna(subset=['commentaire_clean'])
  st.write(nombre_doublon , "lignes doublons supprimées:")

  st.write("Après suppression duplicates", df3.shape)

  st.markdown("<h2 style='text-align: left;'>Le Clusternig:</h2>", unsafe_allow_html=True)

  st.markdown("<h3 style='text-align: left;'>Définition du Clustering:</h3>", unsafe_allow_html=True)

  st.markdown("<p style='text-align: left;'>Le clustering (Le partitionnement de données) consiste à créer des groupes "
              "d'individus de telle sorte que les individus d'un groupe donné aient tendance à être similaires, et en même "
              "temps aient tendance à être différents des individus des autres groupes. Les algorithmes de classification "
              "non supervisée répondent à cette tâche, il recherche les structures naturelles dans les données car les données "
              "cibles sont absentes lors de l’apprentissage. Son objectif est de construire des classes automatiquement en "
              "fonction des instances (ou des observations) disponibles. </p>", unsafe_allow_html=True)
  
  st.markdown("<h3 style='text-align: left;'>Préparation des données:</h3>", unsafe_allow_html=True)

  st.markdown("<p style='text-align: left;'>Notre jeu de données contenait les avis de plusieurs compagnies appartenantes aux différents secteurs d’activités"
              " : secteur bancaire, secteur de l’habillement, secteur de voyage et vacances secteur technologiques… "
               " Nous allons donc isoler une catégorie avant de faire le clusternig </p>", unsafe_allow_html=True)

  # notes__bis = [0,1 ]
  notes__bis = st.sidebar.radio("Choisissez les notes à analyser:", [0,1])

  liste_liens2 = ['bank','mortgage_broker','travel_insurance_company','insurance_agency']
  catégorie_ = st.sidebar.selectbox('Choisissez la catégorie à analyser:', liste_liens2)

  df3 = df3.loc[((df3['notes_bis'] == notes__bis))  &  (df3['categorie_bis'] == catégorie_ ) ]

  st.write("Nombre de lignes de la catégorie: ", df3.shape)

  # st.write('étape 0: Créer un vecteur TF-IDF à partir de la colonne commentaire_clean')
  # Créer un vecteur TF-IDF à partir de la colonne 'commentaire_clean'

  tfidf_vectorizer = TfidfVectorizer(stop_words='english')
  tfidf_matrix = tfidf_vectorizer.fit_transform(df3['commentaire_clean'])
 
  sse = []
  for k in range(2, 8):
      kmeans = KMeans(n_clusters=k, random_state=42)
      kmeans.fit(tfidf_matrix)
      sse.append(kmeans.inertia_)
      
  # sse = joblib.load("models/modele_clusters_lib")

  st.markdown("<h3 style='text-align: left;'>Le nombre de clusters optimal:</h3>", unsafe_allow_html=True)

  st.write("Déterminer le nombre optimal de clusters en utilisant la méthode du coude. "
  "Cela implique d'ajuster le modèle KMeans pour différents nombres de clusters "
  "et de calculer la somme des carrés des distances pour chaque ajustement "
  "puis de choisir le coude du graphique comme nombre optimal de clusters..")

  fig, ax = plt.subplots(figsize=(10,4))
  plt.plot(range(2, 8), sse, marker='o')
  plt.title('Méthode du coude pour déterminer le nombre optimal de clusters')
  plt.xlabel('Nombre de clusters')
  plt.ylabel('Somme des carrés des distances (SSE)')
  st.pyplot(fig)

  # Choisissez le nombre optimal de clusters en fonction du graphique du coude
  
  # st.write("#### Afficher un slider de 1 à 7")
  optimal_k = st.sidebar.slider('Choisir le nombre de cluster:', 1, 8, value = 3)
  
  # optimal_k = 4

  # Appliquer KMeans avec le nombre optimal de clusters
  kmeans = KMeans(n_clusters=optimal_k, random_state=42)
  kmeans.fit(tfidf_matrix)
  # kmeans = joblib.load("models/modele_kmeans1_lib")

  # Ajouter les étiquettes de cluster à votre DataFrame
  df3['cluster'] = kmeans.labels_ + 1

  
  st.markdown("<h3 style='text-align: left;'>Répartition par Cluster</h3>", unsafe_allow_html=True)
  st.markdown("<p style='text-align: left;'>Voici la répartition des avis par clusters </p>", unsafe_allow_html=True)


  # Compter le nombre d'avis par cluster
  cluster_counts = df3['cluster'].value_counts().sort_index()

  # Diagramme en barres
  fig , ax = plt.subplots(figsize=(10,4))
  plt.bar(cluster_counts.index , cluster_counts.values, color='skyblue')
  plt.xticks(cluster_counts.index , labels=cluster_counts.index )
  
  plt.xlabel('Cluster')
  plt.ylabel('Nombre d\'avis')
  plt.title('Distribution des avis par cluster')

  # plt.show()
  st.pyplot(fig)
  st.markdown("<h3 style='text-align: left;'>Afficher les principaux termes par cluster:</h3>", unsafe_allow_html=True)

  # Afficher les principaux termes par cluster
  order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]
  terms = tfidf_vectorizer.get_feature_names_out()
  liste_tab = []
  for i in range(optimal_k):
      top_terms = [terms[ind] for ind in order_centroids[i, :20]]
      st.write(f"**Cluster {i+1}**: {', '.join(top_terms)}")
      # cluster_n =  f"Cluster {i+1}"
      cluster_n =  f"{i+1}"
      # st.write(cluster_n)
      liste_tab.append(cluster_n)

  st.markdown("<h3 style='text-align: left;'>Nuage de mots par cluster:</h3>", unsafe_allow_html=True)
  st.markdown("<p style='text-align: left;'>Après avoir fait ces nuages de points,"
              " nous allons essayer de trouver une logique qui permettra de définir la spécificité de chacun des clusters </p>", unsafe_allow_html=True)

  # for tabss in liste_tab:
  #   tabss = st.tabs(tabss)
  #   st.header(tabss)
  #   st.image("https://static.streamlit.io/examples/cat.jpg", width=200)


  # Sélection de l'onglet
  
  # liste_tab = [int(valeur) for valeur in liste_tab]

  # tab_cluster = st.radio("Choisissez les notes à analyser:", liste_tab)
  # st.write(tab_cluster)

  # st.header("Cluster: ")
  # df3_1 = df3[df3['cluster'] == tab_cluster]
  # # Exemple pour le cluster 0
  # text_cluster_1 = ' '.join(df3_1['commentaire_clean'])
  # # Générer le nuage de mots
  # wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text_cluster_1)
  # # Afficher le nuage de mots
  # fig , ax = plt.subplots(figsize=(10,6))
  # plt.imshow(wordcloud, interpolation='bilinear')
  # plt.axis('off')
  # plt.title('Nuage de mots pour le cluster: ', tab_cluster)
  # st.pyplot(fig)
  
  
  tab1, tab2, tab3 = st.tabs(["Cluster 1", "Cluster 2", "Cluster 3"])
  with tab1:
    st.header("Cluster 01")
    df3_1 = df3[df3['cluster'] == 1]
    # Exemple pour le cluster 0
    text_cluster_1 = ' '.join(df3_1['commentaire_clean'])
    # Générer le nuage de mots
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text_cluster_1)
    # Afficher le nuage de mots
    fig , ax = plt.subplots(figsize=(10,6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Nuage de mots pour le cluster 1')
    st.pyplot(fig)

    # st.image("https://static.streamlit.io/examples/cat.jpg", width=200)

  with tab2:
    st.header("Cluster 02")
    df3_2 = df3[df3['cluster'] == 2]
    # Exemple pour le cluster 0
    text_cluster_2 = ' '.join(df3_2['commentaire_clean'])
    # Générer le nuage de mots
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text_cluster_2)
    # Afficher le nuage de mots
    fig , ax = plt.subplots(figsize=(10,6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Nuage de mots pour le cluster 2')
    st.pyplot(fig)

  with tab3:
    st.header("Cluster 03")
    df3_3 = df3[df3['cluster'] == 3]
    # Exemple pour le cluster 0
    text_cluster_3 = ' '.join(df3_3['commentaire_clean'])
    # Générer le nuage de mots
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text_cluster_3)
    # Afficher le nuage de mots
    fig , ax = plt.subplots(figsize=(10,6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Nuage de mots pour le cluster 3')
    st.pyplot(fig)



  # st.dataframe(df3.head(10))
  





############# les auteurs 
st.sidebar.write("**Réalisé par:**")
st.sidebar.markdown("[Mustapha LAACHIR](https://www.laapha.com)")
st.sidebar.markdown("[Sabrina DIACQUENOD](https://www.laapha.com)")



####################    divers   ##########""
# # if page == pages[5] : 
# if page == 5: 
#   from PIL import Image
#   from io import BytesIO
#   import base64

#   st.write("#### Afficher un slider de 1 à 7")
#   st.slider('Slider', 1, 7)

#   st.write("#### Afficher un slider de Lundi à Dimanche")
#   st.select_slider('Choisir un jour de la semaine', options=
#   ['Lundi','Mardi','Mercredi','Jeudi','Vendredi','Samedi','Dimanche'])

#   st.write("#### Inserer")
#   st.text_input('Insérez votre texte')
#   st.number_input('Choisissez votre nombre')
#   st.date_input('Choisissez une date')             
#   st.time_input('Choisissez un horaire')
#   st.file_uploader('Importer votre fichier')
#   st.code(''' import streamlit ''', language='python')

#   st.markdown("Ceci est un [lien hypertexte](https://www.example.com) vers Example.com.")

#   # st.image(image, caption='C est une image')
           
#   # pip install streamlit-drawable-canvas
#   # streamlit-drawable-canvas()
#   st.sidebar.write("## Upload and download :gear:")

#   ###########test imoprt images....
#   MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB

#   # Download the fixed image
#   def convert_image(img):
#     buf = BytesIO()
#     img.save(buf, format="PNG")
#     byte_im = buf.getvalue()
#     return byte_im

#   def fix_image(upload):
#     image = Image.open(upload)
#     col1.write("Original Image :camera:")
#     col1.image(image)

#     # fixed = remove(image)
#     col2.write("Fixed Image :wrench:")
#     col2.image(image)
#     st.sidebar.markdown("\n")
#     st.sidebar.download_button("Download fixed image", convert_image(image), "fixed.png", "image/png")


#   col1, col2 = st.columns(2)

#   my_upload = st.sidebar.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

#   if my_upload is not None:
#     if my_upload.size > MAX_FILE_SIZE:
#         st.error("The uploaded file is too large. Please upload an image smaller than 5MB.")
#     else:
#         fix_image(upload=my_upload)
#   else:
#     fix_image("./zebra.jpg")

#     ## permets d'afficher une table df et colorier le max en jaune
#   st.write("Avec un dataframe fixe")
#   dataframe = pd.DataFrame(
#     np.random.randn(10, 8),
#     columns=('col %d' % i for i in range(8)))

#   st.dataframe(dataframe.style.highlight_max(axis=0))
#   ## table
#   st.write("Avec une table fixe")
#   dataframe = pd.DataFrame(
#     np.random.randn(10, 20),
#     columns=('col %d' % i for i in range(20)))
#   st.table(dataframe)
  
#   #   st.text_input("Your name", key="name")
#   st.write("You can access the value at any point with key: URL est défini dans une autre page")
#   st.write(" la valeur de url est: \n",st.session_state.URL)

#   col3, col4 = st.columns(2)
#   col3.write("fin col 3")
#   col4.write("fin col 4")



#   ############################## Fonction pour scraper le lien avec BeautifulSoup
#   st.write("### Fonction pour scraper le lien avec BeautifulSoup")
#   # Fonction pour scraper le lien avec BeautifulSoup
#   def scrape_link(url):
#       try:
#           # Obtenir le contenu de la page web
#           response = requests.get(url,verify = False)
#           soup = bs(response.content, 'html.parser')

#           # Ici, vous pouvez ajouter votre logique de scraping en utilisant BeautifulSoup
#           # Par exemple, si vous voulez extraire tous les liens de la page :
#           links = soup.find_all('a')
#           extracted_links = [link.get('href') for link in links]

#           return extracted_links

#       except Exception as e:
#           st.error(f"Une erreur s'est produite : {e}")
#           return []

#   # Interface Streamlit
#   st.title("Scraping avec BeautifulSoup")

#   # Champ de saisie pour l'URL
#   url = st.text_input('Entrez une URL à scraper :', 'https://www.laapha.com')

#   # Bouton pour lancer le scraping
#   if st.button("Scrape"):
#       if url:
#           # Appeler la fonction de scraping
#           scraped_data = scrape_link(url)
          
#           # Afficher les résultats
#           st.write("Liens extraits de la page :")
#           for link in scraped_data:
#               st.write(link)
#       else:
#           st.warning("Veuillez saisir une URL valide.")

st.image("médias/avis_clients.png",  caption='Avis clients',  use_column_width=False, width=50 )