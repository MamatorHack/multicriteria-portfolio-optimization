# 📈 Portfolio Manager Pro

Une application interactive d'optimisation de portefeuille financier basée sur la théorie moderne du portefeuille (Markowitz), intégrant des contraintes réalistes (coûts de transaction et cardinalité).

## 🚀 Fonctionnalités

* **Univers d'Investissement Dynamique :** Sélection d'actifs par secteurs économiques (Tech, Santé, Énergie...) via un fichier de configuration JSON.
* **Gestion de Portefeuille "Stateful" :** Interface robuste permettant de saisir ses allocations actuelles sans rechargement intempestif de la page.
* **Optimisation Multi-Critères :**
    * Maximisation du Rendement ($f_1$)
    * Minimisation du Risque / Volatilité ($f_2$)
    * Minimisation des Coûts de Transaction ($f_3$)
* **Simulation Monte Carlo :** Génération de milliers de portefeuilles pour visualiser la frontière efficiente.
* **Outils d'Aide à la Décision :**
    * Visualisation 3D interactive (Rendement / Risque / Coût).
    * Plan d'arbitrage automatique (Quoi acheter/vendre ?).
    * Projection de fortune sur 10 ans avec intervalle de confiance.

## 📦 Installation

1.  **Cloner le projet :**
    ```bash
    git clone [https://github.com/votre-repo/portfolio-manager.git](https://github.com/votre-repo/portfolio-manager.git)
    cd portfolio-manager
    ```

2.  **Installer les dépendances :**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Vérifier la présence du fichier `tick.json`** à la racine (contient les tickers boursiers).

## ▶️ Utilisation

Lancez l'application via Streamlit :

```bash
streamlit run app.py
````

L'application s'ouvrira automatiquement dans votre navigateur (http://localhost:8501).

## 🗂️ Structure du Code

  * `app.py` : Point d'entrée de l'application. Gère l'interface utilisateur (UI) et la navigation.
  * `modules/` :
      * `data_loader.py` : Gestion du téléchargement des données (API Yahoo Finance) et cache.
      * `optimizer.py` : Moteur mathématique (Simulation Monte Carlo, Calcul des ratios).
      * `plots.py` : Génération des graphiques interactifs (Plotly).

## 🛠️ Technologies

  * **Python 3.9+**
  * **Streamlit** (Interface Web)
  * **Yahoo Finance (yfinance)** (Données de marché)
  * **Plotly** (Visualisation 3D)
  * **NumPy / Pandas** (Calcul matriciel)

## 📝 Auteur

Projet réalisé dans le cadre du cours d'Analyse Multicritère. Et généré en partie via Gemini