# Formation "IA et patrimoine"

Plan : 

- [*Introduction*](#introduction)
- [*Apprentissage supervisé*](#apprentissage-supervisé)
- [*Apprentissage non supervisé*](#apprentissage-non-supervisé)
- [*Ateliers*](#ateliers)

***

# Introduction

## Matrice de confusion

<i>Section "B. Evaluation des résultats" du support de cours</i>

- [github](https://github.com/altomator/Formation_IA/tree/main/iris)
- Source : [Evaluate the performance of a classifier with Confusion Matrix](https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html) (scikit-learn.org)

1. Visualiser le dataset avec [iris_dataset.py ](https://github.com/altomator/Formation_IA/blob/main/iris/iris_dataset.py), en affichant les variables longueur/largeur pour les pétales et les sétales.
   
   ![Visualisation des variables pétale (longueur, largeur)](https://github.com/altomator/Formation_IA/blob/main/iris/petal.png)

> L'espèce Setosa est caractérisée par une petite taille de pétales.
   
3. Calculer la matrice de confusion avec le [notebook](https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html) (utiliser JupyterLite - bouton <b>Launch Lite</b> — ou tout autre application compatible)

   ![Visualisation de la matrice de confusion](https://github.com/altomator/Formation_IA/blob/main/iris/matrice.png)

> Interprétation : confusion du modèle entre les espèces versicolor et virginica.

# Apprentissage supervisé

## Arbre de décision

<i>Section "C. Apprentissage machine" du support de cours</i>

- [github](https://github.com/altomator/Formation_IA/tree/main/arbre)
- Source : [Plot the decision surface of decision trees trained on the iris dataset](https://scikit-learn.org/stable/auto_examples/tree/plot_iris_dtc.html) (scikit-learn.org)

2. Calculer l'arbre de décision avec le [notebook](https://scikit-learn.org/stable/auto_examples/tree/plot_iris_dtc.html) (utiliser JupyterLite)

   ![Visualisation de l'arbre de décision](https://github.com/altomator/Formation_IA/blob/main/arbre/arbre.png)

> Interprétation : dans la première noeud du modèle, la variable x[2] (longueur de pétale) permet de prédire intégralement la classe n° 1 (setosa, cf. graphe ci-avant)
 :
``` x[2] <= 2.45 ```

## Perceptron

<i>Section "C. Apprentissage machine" du support de cours</i>

[Démo web Loria](https://mlweb.loria.fr/book/en/perceptron.html)

[Démo web Tensorflow](https://playground.tensorflow.org)



## Perceptron multicouche (avec keras)

<i>Section "C. Apprentissage machine" du support de cours</i>

[Démo web Tensorflow](https://playground.tensorflow.org)

<b>Atelier Perceptron multicouche :</b>
- [github](https://github.com/altomator/Formation_IA/tree/main/perceptron)
- Source : [Iris Neural Network](https://github.com/damiannolan/iris-neural-network/blob/master/iris-neural-network.ipynb)

1. Télécharger le notebook.
2. Le charger dans Google Colab.
3. Etudier la construction du réseau de neurones. 
4. Produire la matrice de confusion.
   
   ![Visualisation de la matrice de confusion](https://github.com/altomator/Formation_IA/blob/main/perceptron/matrice.png)


##  Réseaux de neurones convolutifs (CNN) 

<i>Section "D. Comprendre les images" du support de cours</i>

[Démo web CNN](https://poloclub.github.io/cnn-explainer/)

## Entrainement d'un CNN avec Kaggle

<b>Atelier :</b>
- Jeu de données : [Iris Computer Vision](https://www.kaggle.com/datasets/jeffheaton/iris-computer-vision/code)
- Notebook : [Iris - TF CNN](https://www.kaggle.com/code/jpmoreux/iris-tf-cnn/)

1. Se connecter sur kaggle.com
2. Ouvrir le [notebook](https://www.kaggle.com/code/jpmoreux/iris-tf-cnn/) 
3. Cliquer sur Copy & Edit notebook
4. Exécuter cellule par cellule
5. Produire et analyser la matrice de confusion.
6. Tracer la courbe de perte  


      ![Visualisation de la perte lors de l'entrainement](https://github.com/altomator/Formation_IA/blob/main/cnn/loss.png)


# Apprentissage non supervisé

## Clusterisation avec K-Means 

<i>Section "C. Apprentissage machine" du support de cours</i>

[Démo web](https://clustering-visualizer.web.app/kmeans)

<b>Atelier :</b> 
- [github](https://github.com/altomator/Formation_IA/tree/main/kmeans)
- Source : [Analyse textuelle avec K-means](https://www.codeandcortex.fr/analyse-textuelle-kmeans/) (Stéphane Meurisse)
- Jeu de données :  [15 000 notices BnF (21e siècle)](https://github.com/altomator/Formation_IA/blob/main/kmeans/21e_15k_titre-auteur-sujet.txt)

1. Créer un environnement Python avec venv
```
python3 -m venv kmeans
source myenv/bin/activate
```

2. Installer les dépendances
```
pip install streamlit bertopic scikit-learn matplotlib pandas sentence-transformers nltk seaborn WordCloud
pip install -U kaleido
```

3. Lancer le script
```
streamlit run clusters.py
```

4. Fournir les données textuelles : [15 000 notices BnF (21e siècle)](https://github.com/altomator/Formation_IA/blob/main/kmeans/21e_15k_titre-auteur-sujet.txt)

- Fichier : ``21e_15k_titre-auteur-sujet.txt``
- Définir le répertoire de travail (local)

5. Cliquer sur l'onglet "Analyse de données"

## Auto-encodeur

[Démo web d'un VAE](https://xnought.github.io/vq-vae-explainer/)

[Démo web d'un VAE]((https://xnought.github.io/vq-vae-explainer/)

***



# Ateliers

## A. Classification d’illustrations 

**Objectif** : entraîner un classifieur permettant de catégoriser 3 types d'illustrations extraites du magazine [_Marie-Claire_](https://gallica.bnf.fr/ark:/12148/cb343488519/date) numérisé dans Gallica :  
- couverture
- double page
- publicité

  ![Illustrations Marie-Claire](https://github.com/altomator/Formation_IA/blob/main/cnn/m-c.png)

Les données à disposition sont les suivantes :
- les fichiers images des illustrations numérisées,
- des données dérivées de ces illustrations :
  - l'OCR extrait des illustrations
  - des descripteurs numériques : taille de l'illustration (en pixels) et nombre de mots de l'OCR
 
**Attendus** :
- Utiliser une des approches suivantes ou toute autre proposition pour classer les types d'illustration.
- Commenter les résultats obtenus.

**Ressources :**
- 170 illustrations : couvertures (30) ; double pages (44) ; publicités (96)
  - [https://github.com/altomator/Formation_IA/tree/main/cnn/marie-claire_img](https://github.com/altomator/Formation_IA/tree/main/marie-claire_img)
- jeux de données dérivées :
  - [données numériques au format CSV](https://github.com/altomator/Formation_IA/tree/main/marie-claire_data)
  - textes océrisés au format texte


   
### Avec Kaggle et Keras : analyse d'images

**Méthode** : entraîner un modèle CNN par transfert learning avec un notebook.

<b>Atelier :</b> 
- Notebook Kaggle de classification d'images d'iris : [Kaggle](https://www.kaggle.com/code/tracyporter/iris-tf-cnn)

_Démarche_ :

0. Ouvrir un compte Kaggle ou se connecter
1. Ouvrir le notebook Kaggle
2. Le copier et l'éditer
3. Importer le dataset d'images de _Marie-Claire_ dans le notebook (Input/Upload)
4. Supprimer le dataset _iris_
5. Adapter le code
6. Evaluer le modèle en ajoutant le calcul de la matrice de confusion ([exemple](https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html))

**Pièges** :

- Lecture des images : attention à l'arborescence des dossiers (`data_dir = ...`)

  ![Courbe de perte durant l'apprentissage](https://github.com/altomator/Formation_IA/blob/main/ateliers/img/cnn-loss.png)

  
![Inférence sur l'ensemble de test](https://github.com/altomator/Formation_IA/blob/main/ateliers/img/cnn-test.png)


### Avec Roboflow : analyse d'images

**Méthode** : entraîner un modèle CNN par transfert learning avec une plateforme IA.

<b>Atelier :</b> 
- https://roboflow.com

_Démarche_ :

0. Créer un compte Roboflow ou se connecter
1. Télécharger le dataset _Marie-Claire_
2. Importer un dossier d’images puis cliquer sur Save and continue
3. Option Label Myself
4. La fenêtre d'annotation s'ouvre, revenir en arrière (on ne veut pas annoter)
5. Cliquer sur Select all
6. puis Actions/Batch Label images et créer la classe correspondant au dossier
7. Ajouter les batchs d’images à un dataset
8. New version (dataset)
9. Ajouter de l’augmentation : Gray, Saturation
10. Entrainer le modèle :
  - Custom
  - Train model : VIT
11. Evaluer le modèle : View Model avec des  images du jeu de test ou des images locales (_upload_)

Option : utiliser le modèle en inférence. Voir exemple [ici (section 4)](https://github.com/altomator/Roboflow). 

**Pièges** :
- Avant l'entrainement, vérifier qu'il n'y a bien que 3 classes dans le dataset Roboflow.
- Performances : la précision est de 100%. Pourquoi ?

 ![Illustrations Marie-Claire](https://github.com/altomator/Formation_IA/blob/main/cnn/roboflow.png)


### Avec Kaggle et SciKit : : analyse de données

**Méthode** : entraîner un modèle à partir de données dérivées numériques.


<b>Atelier :</b> 
- [Jeu de données numériques](https://github.com/altomator/Formation_IA/tree/main/marie-claire_data)
- https://kaggle.com

_Démarche_ :

0. Ouvrir un compte Kaggle ou se connecter
1. Utiliser une de ces approches en s’inspirant des notebooks suivants :  
1.a : SVM avec comme base le notebook [Kaggle](https://www.kaggle.com/code/prashant808/iris-dataset-using-svm))  
1.b : Arbre de décision avec comme base le notebook [Kaggle](https://www.kaggle.com/code/sheemamasood/decisiontree-classifier-iris )
2. Le copier et l'éditer
3. Etudier la forme des données sur les axes largeur, hauteur, nombre de mots (utiliser `matplotlib.pyplot`)
4. Lancer l'entrainement
5. Etudier les performances

**Pièges SVM** :

- Lecture du fichier CSV : le caractère délimiteur est ';'.
- Préparation des données : il faut supprimer les colonnes de données inutiles pour l'entrainement (chemin, orientation...) avec `df.drop()`. `x` doit ne contenir que des valeurs numériques.
- Evaluation du modèle : ajouter le calcul de la matrice de confusion ([exemple](https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html))


	![Matrice de confusion](https://github.com/altomator/Formation_IA/blob/main/ateliers/img/svm-matrix.png)



**Pièges Arbres** :

 ![Visualisation des données](https://github.com/altomator/Formation_IA/blob/main/marie-claire_data/3d.png)


### Avec un LLM : classification de textes

**Méthode** : utiliser un LLM en zero _shot learning_ pour classer les illustrations, connaissant leur texte océrisé.


<b>Atelier :</b> 
- [Textes océrisés](https://github.com/altomator/Formation_IA/tree/main/marie-claire_data/mc-ocr.zip)
- outil de codage Python
- LLM en mode API (Mistral)
- clé : "qT4ZADzNuabqm2JmmT1erkLotuuZATfC"
- exemple Python : https://github.com/altomator/Formation_IA/blob/main/ateliers/exemple.py

_Démarche_ :

1. Utiliser les classes "Double page" et "Publicité" qui peuvent être ambiguës de par leur contenu textuel. La détection de la classe Couverture est triviale, ne pas la traiter (présence systématique du texte "Marie-Claire").
2. Prompter le LLM pour qu'il produise une décision, sa justification et un résumé du texte.
3. Stocker dans un fichier JSON (un par illustration).
4. Evaluer les performances.

Option : utiliser la bibliothèque Pydantic pour valider la sortie JSON du LLM :
- [Mistral](https://docs.mistral.ai/capabilities/structured_output/custom)
- [Pydantic](https://docs.pydantic.dev/latest/concepts/models/)

### BILAN

| Approche  | Performances          | Commentaires |
| :--------------- |---------------:| :-----:|
| CNN   |   82%       |  Performances calculées d'après peu de données (17). Confusion majoritaire entre couvertures et publicités |
| SVM  | 80%             |   Performances calculées d'après peu de données. Confusion systématique entre couvertures et publicités ? |
| Arbre de décision  |           |     |

## B. n8n


