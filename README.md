# Formation IA

# Apprentissage non supervisé

## Matrice de confusion

Atelier :
- [github](https://github.com/altomator/Formation_IA/tree/main/iris)
- source : [Evaluate the performance of a classifier with Confusion Matrix](https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html) (https://scikit-learn.org/)

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



