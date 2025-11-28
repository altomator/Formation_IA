# Formation IA

# Apprentissage non supervisé

## Clusterisation

[Démo web](https://clustering-visualizer.web.app/kmeans)

Atelier K-Means:
- [github](https://github.com/altomator/Formation_IA/tree/main/kmeans)
- source : [Analyse textuelle avec K-means](https://www.codeandcortex.fr/analyse-textuelle-kmeans/) (Stéphane Meurisse)

1. Créer un environnement Python

2. Dépendances
```
pip install streamlit bertopic scikit-learn matplotlib pandas sentence-transformers nltk seaborn WordCloud
pip install -U kaleido
```

3. Script
```
streamlit run clusters.py
```

Fournir les données textuelles : 15 000 notices BnF (21e siècle)

Fichier : ``21e_15k_titre-auteur-sujet.txt``

# Apprentissage supervisé

## Perceptron

[Démo web](https://mlweb.loria.fr/book/en/perceptron.html)



