# Formation IA

# Apprentissage non supervisé

## Matrice de confusion

Atelier :
- [github](https://github.com/altomator/Formation_IA/tree/main/iris)
- source : [Evaluate the performance of a classifier with Confusion Matrix](https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html) (scikit-learn.org)

1. Visualiser le dataset avec [iris_dataset.py ](https://github.com/altomator/Formation_IA/blob/main/iris/iris_dataset.py), en affichant les variables longueur/largeur pour les pétales et les sétales.
   
   ![Visualisation des variables pétale (longueur, largeur)](https://github.com/altomator/Formation_IA/blob/main/iris/petal.png)

   L'espèce Setosa est caractérisée par une petite taille de pétales.
   
3. Calculer la matrice de confusion avec le [notebook](https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html) (utiliser JupyterLite - bouton <b>Launch Lite</b> — ou tout autre application compatible)

   ![Visualisation de la matrice de confusion](https://github.com/altomator/Formation_IA/blob/main/iris/matrice.png)

## Arbre de décision

Atelier :
- [github](https://github.com/altomator/Formation_IA/tree/main/arbre)
- source : [Plot the decision surface of decision trees trained on the iris dataset](https://scikit-learn.org/stable/auto_examples/tree/plot_iris_dtc.html) (scikit-learn.org)

2. Calculer l'arbre de décision avec le [notebook](https://scikit-learn.org/stable/auto_examples/tree/plot_iris_dtc.html) (utiliser JupyterLite)

   ![Visualisation de l'arbre de décision](https://github.com/altomator/Formation_IA/blob/main/arbre/arbre.png)

## Clusterisation

[Démo web](https://clustering-visualizer.web.app/kmeans)

Atelier K-Means :
- [github](https://github.com/altomator/Formation_IA/tree/main/kmeans)
- source : [Analyse textuelle avec K-means](https://www.codeandcortex.fr/analyse-textuelle-kmeans/) (Stéphane Meurisse)

1. Créer un environnement Python

2. Dépendances
```
pip install streamlit bertopic scikit-learn matplotlib pandas sentence-transformers nltk seaborn WordCloud
pip install -U kaleido
```

3. Lancer le script
```
streamlit run clusters.py
```

4. Fournir les données textuelles : 15 000 notices BnF (21e siècle)

Fichier : ``21e_15k_titre-auteur-sujet.txt``

# Apprentissage supervisé

## Perceptron

[Démo web](https://mlweb.loria.fr/book/en/perceptron.html)


## Perceptron multicouche (keras)

Atelier Perceptron multicouche :
- [github](https://github.com/altomator/Formation_IA/tree/main/perceptron)
- source : [Iris Neural Network](https://github.com/damiannolan/iris-neural-network/blob/master/iris-neural-network.ipynb)

1. Télécharger le notebook.

2. Le charger dans Google Colab.
   
3. Produire la matrice de confusion.
   
   ![Visualisation de la matrice de confusion](https://github.com/altomator/Formation_IA/blob/main/perceptron/matrice.png)


