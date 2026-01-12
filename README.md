# Formation "IA et patrimoine"

# Apprentissage machine

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

- [github](https://github.com/altomator/Formation_IA/tree/main/arbre)
- source : [Plot the decision surface of decision trees trained on the iris dataset](https://scikit-learn.org/stable/auto_examples/tree/plot_iris_dtc.html) (scikit-learn.org)

2. Calculer l'arbre de décision avec le [notebook](https://scikit-learn.org/stable/auto_examples/tree/plot_iris_dtc.html) (utiliser JupyterLite)

   ![Visualisation de l'arbre de décision](https://github.com/altomator/Formation_IA/blob/main/arbre/arbre.png)

> Interprétation : dans la première noeud du modèle, la variable x[2] (longueur de pétale) permet de prédire intégralement la classe n° 1 (setosa, cf. graphe ci-avant)
 :
``` x[2] <= 2.45 ```

## Perceptron

[Démo web Loria](https://mlweb.loria.fr/book/en/perceptron.html)

[Démo web Tensorflow](https://playground.tensorflow.org)



## Perceptron multicouche (avec keras)

[Démo web Tensorflow](https://playground.tensorflow.org)

Atelier Perceptron multicouche :
- [github](https://github.com/altomator/Formation_IA/tree/main/perceptron)
- source : [Iris Neural Network](https://github.com/damiannolan/iris-neural-network/blob/master/iris-neural-network.ipynb)

1. Télécharger le notebook.

2. Le charger dans Google Colab.
3. Etudier la construction du réseau de neurones. 
4. Produire la matrice de confusion.
   
   ![Visualisation de la matrice de confusion](https://github.com/altomator/Formation_IA/blob/main/perceptron/matrice.png)


## Analyse d'images : entrainement d'un modèle avec Kaggle

Atelier :
- jeu de données : [Iris Computer Vision](https://www.kaggle.com/datasets/jeffheaton/iris-computer-vision/code)
- script : [Iris - TF CNN](https://www.kaggle.com/code/tracyporter/iris-tf-cnn)

1. Se connecter sur kaggle.com
2. Ouvrir la page du jeu de données Iris
3. Aller sur l'onglet Code
4. Choisir le notebook [Iris - TF CNN](https://www.kaggle.com/code/tracyporter/iris-tf-cnn)
5. Cliquer sur Copy & Edit notebook
6. Exécuter cellule par cellule
7. Produire et analyser la matrice de confusion.
   


# Apprentissage non supervisé

## Clusterisation avec K-Means 

[Démo web](https://clustering-visualizer.web.app/kmeans)

Atelier : 
- [github](https://github.com/altomator/Formation_IA/tree/main/kmeans)
- source : [Analyse textuelle avec K-means](https://www.codeandcortex.fr/analyse-textuelle-kmeans/) (Stéphane Meurisse)

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








# Apprentissage supervisé



