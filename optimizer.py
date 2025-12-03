import numpy as np
import pandas as pd
import os
from scipy.optimize import minimize

class PortfolioOptimizer:
    def __init__(self, data_folder="data"):
        """
        Initialise l'optimiseur en chargeant les données réelles depuis les CSV.
        """
        self.data_folder = data_folder
        self.tickers = []
        self.sectors = []
        self.mu = None
        self.sigma = None
        self.asset_names = []
        
        # Chargement et traitement des données
        self.load_and_process_data()

    def load_and_process_data(self):
        """Lit les CSV, fusionne les données et calcule Mu et Sigma."""
        combined_df = pd.DataFrame()
        sector_map = {}

        # 1. Lecture de tous les CSV dans le dossier data
        if not os.path.exists(self.data_folder):
            raise FileNotFoundError(f"Le dossier '{self.data_folder}' n'existe pas. Lancez download.py d'abord.")

        for filename in os.listdir(self.data_folder):
            if filename.endswith(".csv"):
                sector_name = filename.replace(".csv", "").replace("_", " ")
                file_path = os.path.join(self.data_folder, filename)
                
                # Lecture du CSV
                df = pd.read_csv(file_path, index_col="Date", parse_dates=True)
                
                # On stocke le secteur pour chaque ticker trouvé
                for ticker in df.columns:
                    sector_map[ticker] = sector_name
                
                # Fusion (concaténation sur l'axe des colonnes)
                if combined_df.empty:
                    combined_df = df
                else:
                    combined_df = pd.concat([combined_df, df], axis=1)

        # 2. Nettoyage des données
        # On supprime les colonnes vides ou avec trop de NaN
        combined_df.dropna(axis=1, how='all', inplace=True) 
        combined_df.dropna(inplace=True) # On garde les dates communes à tous

        # 3. Calcul des Rendements (Returns)
        # On utilise le log-returns ou simple pct_change. Ici pct_change pour Markowitz classique.
        returns_df = combined_df.pct_change().dropna()

        # 4. Stockage des attributs
        self.asset_names = list(returns_df.columns)
        self.n_assets = len(self.asset_names)
        
        # Mapping des secteurs pour l'affichage (aligné avec l'ordre des assets)
        self.sectors = [sector_map.get(t, "Unknown") for t in self.asset_names]

        # 5. Calculs Financiers
        # Rendement moyen annuel (252 jours de trading)
        self.mu = returns_df.mean().values * 252
        
        # Matrice de Covariance annuelle
        self.sigma = returns_df.cov().values * 252

    # --- Fonctions Objectifs (Inchangées) ---
    def f1_return(self, w):
        return -np.dot(w, self.mu)

    def f2_risk(self, w):
        return np.dot(w.T, np.dot(self.sigma, w))

    def f3_cost(self, w, w_current, c_prop):
        if w_current is None: return 0
        return np.sum(c_prop * np.abs(w - w_current))

    # --- Moteur d'Optimisation (Inchangé) ---
    def optimize(self, target_return=None, w_current=None, c_prop=0.0, k_cardinality=None, lmbda=0.5):
        n = self.n_assets
        
        def objective(w):
            risk = self.f2_risk(w)
            ret = -self.f1_return(w)
            cost = self.f3_cost(w, w_current, c_prop)
            # Normalisation basique pour éviter que le risque (souvent petit) soit écrasé par le rendement
            # Note: Dans un cas réel, on affine lambda. Ici on garde simple.
            return lmbda * risk - (1 - lmbda) * ret + cost

        cons = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
        
        if target_return is not None:
             cons.append({'type': 'ineq', 'fun': lambda w: -self.f1_return(w) - target_return})

        bounds = [(0, 1) for _ in range(n)]

        # Gestion Cardinalité (Heuristique simplifiée)
        if k_cardinality is not None and k_cardinality < n:
            w0 = np.ones(n) / n
            # Petite optimisation rapide pour trouver les meilleurs candidats
            res_relaxed = minimize(objective, w0, method='SLSQP', bounds=bounds, constraints=cons)
            if res_relaxed.success:
                top_indices = np.argsort(res_relaxed.x)[-k_cardinality:]
                bounds = []
                for i in range(n):
                    if i in top_indices: bounds.append((0, 1))
                    else: bounds.append((0, 0)) 

        w0 = np.ones(n) / n
        result = minimize(objective, w0, method='SLSQP', bounds=bounds, constraints=cons)
        
        if result.success:
            return result.x
        else:
            return None

    def get_portfolio_metrics(self, w, w_current=None, c_prop=0.0):
        ret = -self.f1_return(w)
        risk = self.f2_risk(w)
        cost = self.f3_cost(w, w_current, c_prop)
        return ret, risk, cost
