# Formation "IA et patrimoine"

**Plan** : 

- [*Introduction*](#introduction)
- [*Apprentissage supervisé*](#apprentissage-supervisé)
- [*Apprentissage non supervisé*](#apprentissage-non-supervisé)
- [*Traitement automatique du langage*](#traitement-automatique-du-langage)
- [*Modèles de langue*](#modèles-de-langue)
- [*Modèles multimodaux*](#modèles-multimodaux)
- [*Ateliers*](#ateliers)

***

# Introduction

## Régression linéaire et perte 

<i>Section "A. Apprentissage machine" du support de cours</i>

- Source : [Régression linéaire](https://developers.google.com/machine-learning/crash-course/linear-regression/parameters-exercise?hl=fr) (developers.google.com)

Ajustez les curseurs Poids et Biais pour trouver le modèle linéaire qui minimise la perte MSE (_Mean squared error_).
   
   ![Régression linéaire](https://github.com/altomator/Formation_IA/blob/main/ateliers/img/mse.png)

## Seuil de classification 

<i>Section "A. Apprentissage machine" du support de cours</i>

- Source : [Seuil et matrice de confusion](https://developers.google.com/machine-learning/crash-course/classification/thresholding?hl=fr) (developers.google.com)

1. Faire varier le seuil.
2. Etudier l'effet du seuil sur les vrais et faux positifs et négatifs.
3. Tester les ensembles de données "différencié", "non séparé" et "déséqulibré".
  
      ![Seuil de classification](https://github.com/altomator/Formation_IA/blob/main/ateliers/img/seuil.png)

## Courbe ROC

<i>Section "A. Apprentissage machine" du support de cours</i>

- Source : [Courber ROC et AUC](https://developers.google.com/machine-learning/crash-course/classification/roc-and-auc?hl=fr) (developers.google.com)

1. Faire varier le seuil de classification.
2. Observer la courbe ROC.
  
      ![Seuil de classification](https://github.com/altomator/Formation_IA/blob/main/ateliers/img/roc.png)

   
## Matrice de confusion

<i>Section "B. Evaluation des résultats" du support de cours</i>

<b>Atelier Python : matrice de confusion</b>
- [Github](https://github.com/altomator/Formation_IA/tree/main/iris)
- Source : [Evaluate the performance of a classifier with Confusion Matrix](https://developers.google.com/machine-learning/crash-course/linear-regression/parameters-exercise?hl=fr) (developers.google.com/machine-learning)

1. Visualiser le dataset avec [iris_dataset.py ](https://github.com/altomator/Formation_IA/blob/main/iris/iris_dataset.py), en affichant les variables longueur/largeur pour les pétales et les sétales.
   
   ![Visualisation des variables pétale (longueur, largeur)](https://github.com/altomator/Formation_IA/blob/main/iris/petal.png)

> L'espèce Setosa est caractérisée par une petite taille de pétales.
   
2. Calculer la matrice de confusion avec le [notebook](https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html) (utiliser JupyterLite - bouton <b>Launch Lite</b> — ou tout autre application compatible)

   ![Visualisation de la matrice de confusion](https://github.com/altomator/Formation_IA/blob/main/iris/matrice.png)

> Interprétation : confusion du modèle entre les espèces versicolor et virginica.


# Apprentissage supervisé

## Arbre de décision

<i>Section "C. Apprentissage machine" du support de cours</i>

<b>Atelier Python : arbre de décision</b>
- [github](https://github.com/altomator/Formation_IA/tree/main/arbre)
- Source : [Plot the decision surface of decision trees trained on the iris dataset](https://scikit-learn.org/stable/auto_examples/tree/plot_iris_dtc.html) (scikit-learn.org)

Calculer l'arbre de décision avec le [notebook](https://scikit-learn.org/stable/auto_examples/tree/plot_iris_dtc.html) (utiliser JupyterLite)

   ![Visualisation de l'arbre de décision](https://github.com/altomator/Formation_IA/blob/main/arbre/arbre.png)

> Interprétation : dans la première noeud du modèle, la variable x[2] (longueur de pétale) permet de prédire intégralement la classe n° 1 (setosa, cf. graphe ci-avant)
 :
``` x[2] <= 2.45 ```

## Perceptron

<i>Section "C. Apprentissage machine" du support de cours</i>

1. Etudier pas à pas le processus d'entrainement du Perceptron : [Démo web Loria](https://mlweb.loria.fr/book/en/perceptron.html)

2. Etudier le processus d'entrainement du Perceptron avec les paramètres suivants : [Démo web Tensorflow](https://playground.tensorflow.org)
   
- Problème = classification
- Paramètres :  
  - 2 entrées
  - 0 couche cachée
  - Activation = Linear
- Données :
  - Dataset linéairement séparable (gaussien)
  - Bruit = 0



   ![Perceptron](https://github.com/altomator/Formation_IA/blob/main/perceptron/perceptron.png)


## Perceptron multicouche (avec keras)

<i>Section "C. Apprentissage machine" du support de cours</i>

Etudier le processus d'entrainement d'un Perceptron avec les paramètres suivants : [Démo web Tensorflow](https://playground.tensorflow.org)
- Problème = classification
- Paramètres :
  - Fonction d’activation RELU ou sigmoïde
  - 2 entrées
  - 1 couche cachée, 2 Perceptrons par couche
- Données :
  - concentriques


<b>Atelier Python : Perceptron multicouche </b>
- [github](https://github.com/altomator/Formation_IA/tree/main/perceptron)
- Source : [Iris Neural Network](https://github.com/damiannolan/iris-neural-network/blob/master/iris-neural-network.ipynb)

Entraînement d’un Perceptron multicouche avec Keras : 2 couches cachées

   ![Réseau](https://github.com/altomator/Formation_IA/blob/main/perceptron/reseau.png)

1. Télécharger le notebook.
2. Le charger dans Google Colab.
3. Etudier la construction du réseau de neurones. 
4. Produire la matrice de confusion.
   
   ![Visualisation de la matrice de confusion](https://github.com/altomator/Formation_IA/blob/main/perceptron/matrice.png)


##  Réseaux de neurones convolutifs (CNN) 

<i>Section "D. Comprendre les images" du support de cours</i>

Etudier l'architecture du réseau et la couche de sortie (_softmax_) : [Démo web CNN](https://poloclub.github.io/cnn-explainer/)

Etudier l'architecture du réseau et les processus de convolution : [Démo web CNN](https://tensorspace.org/html/playground/inceptionv3.html)


## Entrainement d'un CNN avec Kaggle

<b>Atelier Python</b>
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

Paramétrage de la [démo web](https://clustering-visualizer.web.app/kmeans) :
- algorithme K-Means
- outil : pinceau
  
![Kmeans](https://github.com/altomator/Formation_IA/blob/main/kmeans/kmeans.png)



## Auto-encodeur

[Démo web d'un VAE : chiffres](https://xnought.github.io/vq-vae-explainer/)

[Démo web d'un VAE : dessin au trait](https://magenta.tensorflow.org/assets/sketch_rnn_demo/index.html)


<br>

# Traitement automatique du langage

## Reconnaissance d'entités nommées

<i>Section "E. Comprendre le langage" du support de cours</i>


[Démo web NER : TextRazor](https://www.textrazor.com/demo)

<b>Atelier Python :</b> 
- Source : [NER avec Random forests et CRF](https://www.kaggle.com/code/jpmoreux/ner-using-crf/)
- Jeu de données :  [Kaggle](https://www.kaggle.com/datasets/abhinavwalia95/entity-annotated-corpus)


1. Se connecter sur kaggle.com
2. Ouvrir le [notebook](https://www.kaggle.com/code/jpmoreux/iris-tf-cnn/) 
3. Cliquer sur Copy & Edit notebook
4. Exécuter cellule par cellule les approches Random Forests puis CRF (Conditional Random Field). Pour CRF, étudier les _features_.

![CRF](https://github.com/altomator/Formation_IA/blob/main/ateliers/img/crf.png)


## Réseaux récurrents

<i>Section "F. Modèles de langue" du support de cours</i>

[Démo web d'entrainement d'un RNN (addition de nombres) : TextRazor](https://storage.googleapis.com/tfjs-examples/addition-rnn/dist/index.html)

[Démo web d'entrainement d'un LSTM](https://storage.googleapis.com/tfjs-examples/lstm-text-generation/dist/index.html)


# Modèles de langue
<i>Section "F. Modèles de langue" du support de cours</i>

## Plongement de mots

[Démo web de word embedding](https://www.cs.cmu.edu/~dst/WordEmbeddingDemo)

[Démo web de word embedding](https://projector.tensorflow.org/)


## Plongement de texte et clusterisation

<i>Sections C. et F. du support de cours</i>

<b>Atelier Python :</b> 
- [Github](https://github.com/altomator/Formation_IA/tree/main/kmeans)
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

## Unités lexicales (token)

[Démo web d'un tokenizer](https://platform.openai.com/tokenizer)

## Modèle BERT

[Démo web BERT](https://cogcomp.seas.upenn.edu/page/demo_view/BERT)

## Température

[Démo web](https://fr.vittascience.com/ia/text)


# Modèles multimodaux

<i>Section "G. LLM et VLM" du support de cours</i>

[Démo web SAM](https://aidemos.meta.com/segment-anything/gallery)

[Démo web OpenCLIP](https://meru.robots.ox.ac.uk/gallica)

[Démo web Gemini](https://gemini.google.com/)

[Démo web OCR Arena](https://www.ocrarena.ai/battle)


<br>
<br>

***
***



# Ateliers

## A. Classification d’illustrations 

**Objectif** : entraîner un classifieur permettant de catégoriser trois types de pages extraites du magazine [_Marie-Claire_](https://gallica.bnf.fr/ark:/12148/cb343488519/date) numérisé dans Gallica :  
- page de couverture
- double page éditoriale
- page de publicité

  ![Illustrations Marie-Claire](https://github.com/altomator/Formation_IA/blob/main/cnn/m-c.png)

Les données à disposition sont les suivantes :
- les fichiers images des pages numérisées,
- des données dérivées de ces pages :
  - des descripteurs numériques : taille de l'image (en pixels) et nombre de mots de l'OCR de la page
  - l'OCR extrait des pages

 
**Attendus** :
- Utiliser une des approches suivantes ou toute autre proposition pour classer les types d'illustration.
- Commenter les résultats obtenus.

**Ressources :**
- 170 illustrations : couvertures (30) ; double pages (44) ; publicités (96)
  - [https://github.com/altomator/Formation_IA/tree/main/cnn/marie-claire_img](https://github.com/altomator/Formation_IA/tree/main/marie-claire_img)
- jeux de données dérivées :
  - [données numériques au format CSV](https://github.com/altomator/Formation_IA/tree/main/marie-claire_data)
  - [textes océrisés au format JSON](https://github.com/altomator/Formation_IA/tree/main/marie-claire_data)


   
### A.1 — Avec Kaggle et Keras : analyse d'images

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


### A.2 — Avec Roboflow : analyse d'images

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


### A.3 — Avec Kaggle et SciKit : analyse de données

**Méthode** : entraîner un modèle à partir des données dérivées numériques (largeur, hauteur, nombre de mots).

<b>Atelier :</b> 
- [Jeu de données numériques](https://github.com/altomator/Formation_IA/tree/main/marie-claire_data)
- https://kaggle.com

 ![Visualisation des données](https://github.com/altomator/Formation_IA/blob/main/marie-claire_data/3d.png)


_Démarche_ :

0. Ouvrir un compte Kaggle ou se connecter
1. Utiliser une de ces approches en s’inspirant des notebooks suivants :  
1.a : SVM avec comme base le notebook [Kaggle](https://www.kaggle.com/code/prashant808/iris-dataset-using-svm))  
1.b : Arbre de décision avec comme base le notebook [Kaggle](https://www.kaggle.com/code/sheemamasood/decisiontree-classifier-iris )
2. Le copier et l'éditer
3. Etudier la forme des données sur les axes largeur, hauteur, nombre de mots (utiliser `matplotlib.pyplot`)
4. Lancer l'entrainement
5. Etudier les performances avec le calcul de métriques adaptées

**Pièges SVM** :

- Lecture du fichier CSV : le caractère délimiteur est ';'.
- Préparation des données : il faut supprimer les colonnes de données inutiles pour l'entrainement (chemin, orientation...) avec `df.drop()`. `x` doit ne contenir que des valeurs numériques. Il faut aussi supprimer la colonne cible (`y`, colonne 'classes')
- Evaluation du modèle : ajouter le calcul de la matrice de confusion ([exemple](https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html))

	![Matrice de confusion](https://github.com/altomator/Formation_IA/blob/main/ateliers/img/svm-matrix.png)


**Pièges Arbres** :
- Lecture du fichier CSV : le caractère délimiteur est ';'.
- Préparation des données : il faut supprimer les colonnes de données inutiles pour l'entrainement (chemin, orientation...) avec `df.drop()`. `x` doit ne contenir que des valeurs numériques. Il faut aussi supprimer la colonne cible (`y`, colonne 'classes')
  

	![Matrice de confusion](https://github.com/altomator/Formation_IA/blob/main/ateliers/img/tree.png)


### A.4 — Avec un LLM : classification de textes

**Méthode** : utiliser un LLM en zero _shot learning_ pour classer les images, connaissant leur texte océrisé.

Notes : 
- La détection de la classe Couverture est triviale, ne pas la traiter (présence systématique du texte "Marie-Claire"). La confusion "couverture" et "publicité" des méthodes précédentes serait donc résolue. 
- Par contre, les double page éditoriales et les publicités peuvent être ambiguës de par leur contenu textuel, certaines éditoriaux présentant des produits commerciaux.

<b>Atelier :</b> 
- [Textes océrisés](https://github.com/altomator/Formation_IA/tree/main/marie-claire_data/mc-ocr.zip)
- outil de codage Python
- LLM en mode API (voir les modèles Mistral [ici](https://docs.mistral.ai/getting-started/models))
- clé d'API Mistral fournie
- exemple de code Python d'appel d'un modèle [Mistral](https://github.com/altomator/Formation_IA/blob/main/ateliers/exemple.py)

_Démarche_ :

1. Utiliser les fichiers OCR des dossiers "publicité" et "éditorial" ou faire des appels à l'API Gallica IIIF Image.
2. Prompter le LLM pour qu'il produise une décision ("publicité" ou "éditorial"), sa justification et un résumé du texte.
3. Stocker la sortie du modèle dans un fichier JSON (un par fichier texte).
4. Stocker la décision du modèle dans un fichier CSV (une ligne par fichier), avec le nom du fichier et le nom du répertoire.
5. Evaluer les performances à la fin du script, en lisant le fichier CSV. Il faut comparer la décision du modèle et le nom du dossier.
6. Evaluer l'influence du prompt sur les performances.

Appel du script (exemple) :
```
>python extract_genre.py publicité/
```
Sortie du script (exemple pour un dossier "publicité") :
```
...
bpt6k4701039n_14.txt == publicité  GREAT!
bpt6k47010424_58.txt == publicité  GREAT!
#bpt6k47013336_12.txt should be 'publicité'#
bpt6k4701076x_45.txt == publicité  GREAT!
---------------------
93 correct predictions out of 96 predictions
Accuracy: 96.88%
```

_Option_ : 
- Utiliser la bibliothèque Pydantic pour valider la sortie JSON du LLM :
  - [Mistral](https://docs.mistral.ai/capabilities/structured_output/custom)
  - [Pydantic](https://docs.pydantic.dev/latest/concepts/models/)

### A.5 — Avec un VLM : classification avec un modèle texte-image

**Méthode** : utiliser un VLM en zero _shot learning_ pour classer les images.

Variante de l'approche précédente :
1. Utiliser les fichiers images des 3 dossiers "couverture, "publicité" et "éditorial".
2. Prompter le VLM pour qu'il produise une décision et sa justification.
3. Stocker la sortie du modèle dans un fichier JSON (un par fichier texte).
4. Stocker la décision du modèle dans un fichier CSV (une ligne par fichier), avec le nom du fichier et le nom du répertoire.
5. Evaluer les performances à la fin du script, en lisant le fichier CSV. Il faut comparer la décision du modèle et le nom du dossier.
6. Evaluer l'influence du prompt sur les performances.

Sortie du script (exemple pour un dossier "publicité") :
```
----------------------
Processing file 89:  bpt6k47011375-PAG_34_IL000001 - Grande.jpeg
...writing in output_file:  ocr_output/bpt6k47011375_34.json
... calling IIIF info    
https://openapi.bnf.fr/iiif/image/v3/ark:/12148/bpt6k47011375/f34/full/,1400/0/default.jpg
...calling Mistral pixtral-large-2411
Decision: advertisement
{
  "category": "advertisement",
  "reasoning": "The image features a promotional layout with marketing arguments such as discounts ('SA CARTE DE RÉDUCTION FAMILIALE'), product pricing ('2,375 Frs', '1,995 Frs'), and a contest announcement ('Participez au GRAND CONCOURS de L'ENCAUSTIQUE SULTANE 100,000 Fr. de prix'). Additionally, it includes a brand logo ('Lévitan'), product images (furniture), and a call to action ('Catalogue gratuit sur demande'), all of which are typical elements of an advertisement."
}
```

### Bilan des performances

Notes :
- jeu de données de petite taille
- Performances parfois calculées d'après peu de données
- Grande sensibilité des LLM et VLM à la forme du prompt

| Approche  | Performances          | Commentaires |
| :--------------- |---------------:| :-----:|
| CNN   |   82%       |  . Confusion entre couvertures et publicités |
| SVM  | 76%             |   Performances calculées d'après peu de données (17). Confusion entre couvertures et publicités |
| Arbre de décision  | 82%          | Performances calculées d'après peu de données (17). Confusion entre couvertures et publicités    |
| LLM  |   éditorial : 86% / publicité :  97%  /       |  Modèle pixtral-large.  Editorial : 38/44 ; Publicité : 93/96 ;  |
| VLM  |   couverture : 100% / éditorial : 82% / publicité : 97%  |  Modèle pixtral-large. Couverture : 30/30 ; Editorial : 36/44 ; Publicité : 93/96. Confusion principale entre éditorial et publicités |


<br>

***

## B. Workflow IA avec n8n

### B.1 — Veille technologique

Utiliser le template : 

### B.1 — Reproduire l'atelier de classification de textes A.4

Utiliser le template : 
