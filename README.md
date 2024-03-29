# Supply_chain_juin23

Ce projet a été réalisé dans le cadre de notre formation de Data scientist réalisée au sein de l'organisme de formation Datascientest.
réalisé par :
- M. LAACHIR 
- S. DIACQUENOD

##1)	Contexte

La supply chain représente les étapes d'approvisionnement allant du processus productif à la distribution de la marchandise au client. 
Suite à ces différentes étapes, la satisfaction client est évaluée afin de : 

-	Étudier la qualité de la supply chain (ex : problème de conception, livraison, prix non adapté, durabilité…)
-	Étudier si le produit/service correspond bien à l’attente du marché.
-	Synthétiser les feedback, améliorations des clients.
-	Aider à la réponse ou à la redirection des clients insatisfaits...
 
Pour de nombreux produits/services, la satisfaction des clients se mesure grâce aux commentaires, et avis laissés par les clients sur des sites dédiées (ex : Trustpilot).

En effet, aujourd’hui un grand nombre de consommateurs consultent activement les avis en ligne avant de faire un achat. Les sites d’avis clients sont des points de rencontre pour ces acheteurs en quête de confiance et de transparence. Les entreprises qui comprennent l'importance de cette dynamique peuvent tirer parti de ces sites pour renforcer leur présence marketing. En affichant des avis authentiques, elles peuvent gagner la confiance des clients potentiels, démontrer la qualité de leurs produits ou services, et influencer positivement les décisions d'achat.

En étudiant les commentaires et avis laissés par les clients, cela va permettre de mieux comprendre les besoins et préférences de ces derniers ; l’objectif étant d’améliorer l’expérience client et éventuellement de fidéliser celui-ci. 

Bien que ce soit important à la compréhension de la satisfaction client., cela peut être long et fastidieux de lire et analyser les verbatim.
L’analyse prédictive de la satisfaction client permise par le Machine Learning aura pour but d’automatiser et donc de simplifier cette tâche. Cela permettra donc une analyse accélérée et plus approfondie des commentaires et avis des clients.

##2)	Objectifs

L’objectif de ce projet est d’extraire de l’information de commentaires laissés par les clients. 
Dans un premier temps, l’objectif sera de prédire la satisfaction d’un client à partir des commentaires laissés, c'est-à-dire de prédire le nombre d'étoiles ou la note donnée à partir des commentaires. 
Puis dans un second temps, l’objectif sera d’extraire les propos du commentaire (problème de livraison, article défectueux...) afin d’expliquer la note attribuée. 
Enfin, l’objectif sera d’extraire de la réponse du fournisseur les propos du commentaire dans le but d’essayer de les prédire uniquement avec le commentaire afin de générer des réponses automatiques. 

##3)	Méthodologie

Nous allons procéder de la manière suivante:
Premièrement, nous allons collecter des données issues du site “Trustpilot”. Ce site répertorie les commentaires et avis clients issus de marques et de domaines différents.

Ensuite, nous procéderons au nettoyage et pré-traitement des données afin d’obtenir un jeu de données propre. 
Puis, nous passerons à la phase de modélisation en sélectionnant un ou plusieurs algorithmes que nous entraînerons sur un ensemble d'entraînement et de test. A l’issue de cette étape, il faudra choisir l’algorithme d'apprentissage automatique le plus performant et évaluer ses performances sur l'ensemble de test en mesurant des métriques telles que la précision, le rappel, la F-mesure, etc.

La dernière étape consistera à intégrer notre modèle dans le processus décisionnel. Une fois que le modèle sera entraîné et validé, il faudra l’intégrer dans le processus de gestion de la supply chain. Il sera utilisé afin d’automatiser l'analyse des commentaires clients, identifier les problèmes potentiels, et même aider à la réponse aux clients insatisfaits de manière plus rapide et efficace.

le projet est présenté dans le streamlit ici: https://supply-chain-reviews.streamlit.app/


