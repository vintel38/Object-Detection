# Utilisation de YoloV4 pour la détection d'objets

Ce répertoire rassemble les fichiers nécessaires pour l'entraînement et l'utilisation de l'algorithme de détection d'objet YoloV4. Il est initialement forké depuis le répertoire d'[Aleksey Bochkovskiy](https://github.com/AlexeyAB/darknet) qui a travaillé et développé la quatrième version de ce programme avec d'autres chercheurs taïwanais. Il s'agit à ce jour d'une des versions les plus performantes d'object detection avec des temps et des cadences de traitements d'images parmi les plus rapides sur les différentes plateformes. Pour rappel, l'object detection est une solution dans le domaine de l'Apprentissage Automatique (Machine Learning) qui permet à un programme de localiser sur un flux vidéo différents objets sur une image à un rythme qui dépend de la fluidité dudit programme. L'intérêt de cette implémentation réside également dans la simplicité du langage Python et la lisibilité des lignes de code des notebooks. Une grande partie des notebooks utilisés dans ce répertoire provient également de la chaine [TheAIGuy](https://www.youtube.com/channel/UCrydcKaojc44XnuXrfhlV8Q) qui a largement démocratisé le répertoire YoloV4.

Pour être clair d'emblée, le but de ce répertoire comme première expérience dans l'object detection est de détecter les objets guitare et personne dans le clip vidéo [Nothing Else Matters](https://www.youtube.com/watch?v=tAGnKpE4NCI) de Metallica. 

Dans un premier temps, je m'intéresserai au processus d'entraînement d'un modèle sur le cloud de Google Colab pour disposer de GPU puissants. Dans un second temps, j'entraînerai un modèle de détection similaire sur une machine portable à savoir un ASUS Zenbook [UX450F](https://www.asus.com/Laptops/For-Home/ZenBook/ZenBook-Pro-14-UX450/) équipé d'une modeste carte graphique GTX1050 MaxQ. La procédure est similaire pour les deux entraînements, mais tout le management de fichiers est effectué au niveau du cloud sur Google Colab ce qui facilite la démarche. Je me concentrerai sur la construction d'un nouveau modèle de détection : en effet, le modèle actuel de Darknet est très performant sur la détection des objets basiques du quotidien mais peu intéressant en l'état pour des applications personnalisées. 
 

<center><img src="https://miro.medium.com/max/320/1*vHWIzPbxmKQSZC6fOyK8Ug.gif" ...></center>
<center>Image de la publication d'A. Bochkovskiy</center>

## 1. Entrainer dans le cloud : Google Colab

[![colab](https://user-images.githubusercontent.com/4096485/86174097-b56b9000-bb29-11ea-9240-c17f6bacfc34.png)](https://colab.research.google.com/drive/1_GdoqCJWXsChrOiY8sZMr_zbr_fH-0Fg)

Avec un compte Google il est possible de réaliser la phase d'entraînement longue et fastidieuse drirectement dans le cloud. Le notebook développé par TheAIGuy vous explique rapidement comment préparer votre dataset pour l'entraînement au travers de différentes étapes. Cependant, le notebook rassemble plusieurs utilisations de l'object detection et pour la présente utilisation, nous n'en utiliserons qu'une partie. L'objectif de cette partie est de se servir de l'algorithme YoloV4 pour entraîner un modèle de détection de guitare sur flux vidéo en effectuant du Transfer Learning depuis les poids pré-entraînés Darknet. 

- Cliquer sur le badge Train in Colab plus haut pour entrer dans le notebook sur la plateforme Google Colab. Commencer par copier le notebook dans votre drive avec l'onglet disponible dans le haut de la fenêtre. Une fois celui-ci retrouvé dans votre drive, placer le dans un dossier dédié pour plus d'ergonomie par la suite. 

- Exécuter les étapes 1 à 5 conformément au chapitre `Running a YOLOv4 Object Detector with Darknet in the Cloud! (GPU ENABLED)` du notebook. Elles vous permettent successivement d'activer l'accélération machine GPU des serveurs de Google, de mettre en place l'architecture YoloV4 ainsi que les poids Darknet qu'il contient pour effectuer une première détection. Des fonctions utiles pour la suite du notebook sont définies également. Les poids Darknet téléchargés ont été préalablement entraînés sur le dataset [COCO](https://cocodataset.org/#home).

- L'étape 6 du notebook permet d'établir un pont entre le serveur de Google et la machine locale (le PC portable). Plusieurs méthodes sont disponibles mais la plus utile pour la quantité de fichiers à télécharger à venir est de connecter le répertoire Google Drive au serveur via la méthode 2. Ainsi, le serveur peut directement interagir avec le contenu du Google Drive personnel et transférer un fichier peut être effectué à l'aide d'une seule ligne de code. 

- Les étapes suivantes ne concernent que les personnes voulant détecter des objects sur images ou vidéos avec les poids déjà entraînés sur les 80 classes du jeu de données [COCO](https://cocodataset.org/#home). Passer au chapitre `How to Train Your Own YOLOv4 Custom Object Detector!` du notebook. Cette partie du notebook nécessite en tout et pour tout 4 éléments qui seront réalisés dans la suite : un jeu d'images annoté, un fichier de configuration `.cfg`, deux fichiers `obj.data` et `obj.names` et deux fichiers textes `train.txt` et `test.txt`.

- Comme dans toute application de Machine Learning, il convient de rassembler et d'étiqueter un jeu de données cohérent avec l'application recherchée. Une adresse utile sur internet pour cette tâche est l'[Open Image Dataset](https://storage.googleapis.com/openimages/web/index.html) de Google qui contient plus de 6 millions d'images annotées tant pour la détection d'objet que pour la segmentation d'image. On peut utiliser l'explorer pour voir que la classe `guitar` existe dans le dataset sans qu'elle soit répertoriée dans les classes du dataset d'entraînement COCO. 

- Pour télécharger les images du dataset de Google, on utilise l'utilitaire [OIDV4](https://github.com/theAIGuysCode/OIDv4_ToolKit) modifié par TheAIGuy pour convenir aux usages de YoloV4. Cloner ce répertoire dans un dossier LOCAL de votre machine (sinon Google Drive va vouloir télécharger individuellement toutes les images que vous allez télécharger du dataset ce qui peut mener à une grosse perte de temps). Avec un invité de commande dans ce répertoire, lancer `pip install -r requirements.txt` qui permet de vérifier que les packages nécessaires sont bien installés dans l'environnement virtuel de travail (dans les bibliothèques de Python si [Anaconda](https://vintel38.github.io/2021/03/28/Anaconda/) n'est pas utilisé). 

- Télécharger les images s'effectue toujours en ligne de commande. 

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`python main.py downloader --classes Guitar --type_csv train --limit 1500`

Cette ligne de commande permet de télécharger 1500 photos contenant la classe guitare. Pour le cas d'une détection multiclasse, on pourra ajouter des étiquette d'images à télécharger derrière `Guitar`. La syntaxe ` --multiclasses 1` sera ajoutée dans le cas d'un téléchargement multiclasses pour que toutes les images soient placées dans un même répertoire.  Afin de disposer d'un dataset complet pour l'entraînement, il est intéressant de télécharger aussi un jeu d'images pour la validation du processus avec une ligne de commande similaire. Pour un certain nombre d'images pour l'entraînement, on utilise généralement 20 à 25% d'images de validation d'où le nombre de 300 pour la ligne de code de validation. 

&nbsp;&nbsp;`python main.py downloader --classes Guitar --type_csv validation --limit 300 `

<center><img src="https://github.com/vintel38/Object-Detection/blob/master/OIDV4.png" ...></center>
<center>Capture d'écran de l'utilitaire OIDv4</center> 

- Créer ensuite un fichier `classes.txt` avec autant de lignes que de classes à détecter. Sur chaque ligne, recopier la classe des images. Utiliser la commande `python convert_annotations.py` pour convertir les annotations issues de OIDv4 en annotations Yolo. Puis dans les dossiers `OIDv4/train/$classe` et `OIDv4/validation/$classe`, supprimer les dossiers `Label`. Vous avez maintenant un dataset prêt à l'emploi pour entraîner votre CNN Yolov4 !


- Dans le dossier `OIDv4/train`, renommer le dossier `$classe` en `obj` et dans le dossier `OIDv4/validation`, renommer le dossier `$classe` en `test`. Compresser ensuite ces deux dossiers en deux archives `obj.zip` et `test.zip` pour préparer l'envoi du dataset sur Google Drive. Déplacer ces deux archives dans votre répertoire Google Drive associé à votre projet. Suivre ensuite les instructions du notebook pour uploader et extraire les données des archives zipées. 

- Il nous faut maintenant produire des fichiers de configuration pour spécifier les entrées et sorties et quelques variables internes au processus d'entraînement. Télécharger et ouvrir le fichier `yolov4-obj.cfg` avec les commandes du notebook. Remplir les variables du fichier de configuration avec des valeurs compatibles avec la solution recherchée. Puis retélécharger ce fichier de configuration sur la session Google Colab. Les deux derniers fichiers de configuration sont facilement réalisables avec les informations du notebook. A noter qu'il ne faut pas oublier de créer le répertoire de backup au bon endroit. A défaut, le processus d'entraînement s'arrêtera lors de la première sauvegarde.

- Recopier depuis ce [répertoire](https://github.com/theAIGuysCode/YOLOv4-Cloud-Tutorial/tree/master/yolov4) le code des fichiers `generate_train.py` et `generate_test.py` puis le transformer en deux fichiers Python à uploader sur la session Google Colab. Exécuter ensuite ces deux scripts qui vont générer	deux fichiers `train.txt` et `test.txt` qui contiendront les chemins respectifs vers les images décompressées et leur étiquettes dans la session Google Colab.  

- La prochaine ligne de code du notebook télécharge les poids Darknet préentraînés et les extrait dans l'architecture Yolov4. 

- Le moment tant attendu est finalement arrivé ! La prochaine ligne de code permet de lancer l'entraînement. Se servir du hack en HTML pour garder la session Google Colab active jusqu'à la fin du nombre d'itérations prévues. 