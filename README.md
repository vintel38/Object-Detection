# Utilisation de YoloV4 pour la détection d'objets

Ce répertoire rassemble les fichiers nécessaires pour l'entraînement et l'utilisation de l'algorithme de détection d'objet YoloV4. Il est initialement forké depuis le répertoire d'[Aleksey Bochkovskiy](https://github.com/AlexeyAB/darknet) qui a travaillé et développé la quatrième version de ce programme avec d'autres chercheurs taïwanais. Il s'agit à ce jour d'une des versions les plus performantes d'object detection avec des temps et des cadences de traitements d'images parmi les plus rapides que ce soit sur un calculateur, sur ordinateur portable ou encore sur système embarqués. Pour rappel, l'object detection est une solution dans le domaine de l'Apprentissage Automatique (Machine Learning) qui permet à un programme de localiser sur un flux vidéo différents objets sur une image à un rythme qui dépend de la fluidité dudit programme. L'intérêt de cette implémentation réside également dans la simplicité du langage Python et 
la lisibilité des lignes de code des notebooks. 

Dans un premier temps, je m'intéresserai au processus d'entraînement d'un modèle sur le cloud de Google Colab pour disposer de GPU puissants. Dans un second temps, j'entraînerai un modèle de détection similaire sur une machine portable à savoir un ASUS Zenbook [UX450F](https://www.asus.com/Laptops/For-Home/ZenBook/ZenBook-Pro-14-UX450/) équipé d'une modeste carte graphique GTX1050 MaxQ. La procédure est similaire pour les deux entraînements, mais tout le management de fichiers est effectué au niveau du cloud sur Google Colab ce qui facilite la démarche. Je me concentrerai sur la construction d'un nouveau modèle de détection : en effet, le modèle actuel de Darknet est très performant sur la détection des objets basiques du quotidien mais peu intéressant en l'état pour des applications personnalisées. 
 

<center><img src="https://miro.medium.com/max/320/1*vHWIzPbxmKQSZC6fOyK8Ug.gif" ...></center>
<center>Image de la publication d'A. Bochkovskiy</center>

## 1. Entrainer dans le cloud : Google Colab

[![colab](https://user-images.githubusercontent.com/4096485/86174097-b56b9000-bb29-11ea-9240-c17f6bacfc34.png)](https://colab.research.google.com/drive/1_GdoqCJWXsChrOiY8sZMr_zbr_fH-0Fg)

Avec un compte Google (ou non), il est possible de réaliser la phase d'entraînement ongue et fastidieuse drirectement dans le cloud. Le notebook vous explique rapidement comment préparer votre dataset pour l'entraînement au travers de différentes étapes.

Lancer une détection avec YoloV4 et le modèle Darknet sur le Cloud :

1. Connecter votre environnement à un GPU pour profiter de l'accélération machine. 
2. Télécharger la structure du réseau de neurones Darknet 
3. Télécharger les poids et biais (weights and biases) du réseau Darknet déjà entraîné sur la database [COCO](https://cocodataset.org/#home) avec 80 classes
4. Définir quelques fonctions utiles pour l'affichage, le téléchargement des données sur le cloud 
5. Lancer une détection pour une image préselectionnée du package 
6. Télécharger et détecter des images sur le cloud depuis différents répertoires
7. Télécharger et détecter des vidéos sur le cloud depuis différents répertoires
8. Customiser les sorties de YoloV4 avec différents drapeaux
9. Détecter plusieurs images en même temps

Entraîner un détecteur YoloV4 customisé :

1. Rassembler et étiqueter les données d'entraînement 
2. Uploader les données d'entraînement sur le cloud 
3. Configurer les fichiers de données pour l'entraînement 
4. Télécharger les poids Darknet préentraînés pour le réseau de neurones
5. Entraîner le réseau de neurones dans le cloud 
6. Vérifier la précision du modèle
7. Lancer la détection d'objet avec le réseau de neurones customisé 