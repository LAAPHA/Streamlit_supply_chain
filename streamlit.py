import streamlit as st
import pandas as pd 
####
data = {
    'Colonne1': [1, 2, 3, 4, 5],
    'Colonne2': ['a', 'b', 'c', 'd', 'e'],
    'Colonne3': [10.1, 20.2, 30.3, 40.4, 50.5],
    'Colonne4': ['x', 'y', 'z', 'w', 'u'],
    'Colonne5': [True, False, True, False, True]
}

# Créer un DataFrame à partir du dictionnaire
df = pd.DataFrame(data)
#####
st.title("Mon premier Streamlit")
st.write("Introduction")
if st.checkbox("Afficher"):
  st.write("Suite du Streamlit")
else:
  st.write("Autre choix")

st.header("Mon deuxième titre")
st.subheader("*Mon troixième titre*") 
st.markdown("affiche du texte au format markdown") 
st.code("affiche du code") 
st.dataframe(df.head(3)) 

st.button("Reset", type="primary")

if st.button('Say hello'):
  st.write('Why hello there')
else:
  st.write('Goodbye')


# st.selectbox() : crée une boite avec différentes options pour obtenir l'affichage sélectionné
# st.slider() : crée un curseur de défilement permettant de sélectionner une valeur numérique parmi une plage donnée
# st.select_slider() : crée un curseur de défilement permettant de sélectionner une valeur non-numérique parmi une plage donnée