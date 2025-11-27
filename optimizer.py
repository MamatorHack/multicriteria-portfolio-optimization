import numpy as np
from scipy.optimize import minimize

class PortfolioOptimizer:
    def __init__(self, n_assets=10):
        # Simulation de données (Remplaçable par yfinance pour des données réelles)
        np.random.seed(42)
        # Rendements attendus (mu) et Matrice de covariance (sigma)
        self.mu = np.random.randn(n_assets) * 0.05 + 0.05
        # Création d'une matrice semi-définie positive pour la covariance
        A = np.random.randn(n_assets, n_assets)
        self.sigma = np.dot(A, A.transpose())
        self.n_assets = n_assets
        self.asset_names = [f"Actif {i+1}" for i in range(n_assets)]
        # Attribution de secteurs aléatoires pour le livrable final
        self.sectors = np.random.choice(["Tech", "Santé", "Finance", "Industrie", "Énergie"], n_assets)

    # --- Fonctions Objectifs [cite: 75, 87] ---
    def f1_return(self, w):
        """Maximiser le rendement (minimiser l'opposé)"""
        return -np.dot(w, self.mu)

    def f2_risk(self, w):
        """Minimiser la variance (Risque de Markowitz)"""
        return np.dot(w.T, np.dot(self.sigma, w))

    def f3_cost(self, w, w_current, c_prop):
        """Coûts de transaction proportionnels: sum(c * |w_new - w_old|)"""
        if w_current is None: return 0
        return np.sum(c_prop * np.abs(w - w_current))

    # --- Moteur d'Optimisation ---
    def optimize(self, target_return=None, w_current=None, c_prop=0.0, k_cardinality=None, lmbda=0.5):
        """
        Résout le problème d'optimisation.
        Gère le Niveau 1 (Markowitz) et Niveau 2 (Cardinalité + Coûts).
        """
        n = self.n_assets
        
        # Fonction objectif combinée (Scalarisation)
        # On minimise: lambda * Risque + (1-lambda) * (-Rendement) + Coûts
        def objective(w):
            risk = self.f2_risk(w)
            ret = -self.f1_return(w) # On remet en positif pour le calcul
            cost = self.f3_cost(w, w_current, c_prop)
            
            # Formule de compromis pondéré
            return lmbda * risk - (1 - lmbda) * ret + cost

        # Contraintes de base [cite: 106]
        # 1. Somme des poids = 1
        cons = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
        
        # 2. Rendement minimal (Si demandé par l'utilisateur via Streamlit) [cite: 124]
        if target_return is not None:
             cons.append({'type': 'ineq', 'fun': lambda w: -self.f1_return(w) - target_return})

        # Bornes: Poids entre 0 et 1 (Pas de vente à découvert) [cite: 107]
        bounds = [(0, 1) for _ in range(n)]

        # --- Gestion de la Cardinalité (Heuristique) [cite: 110] ---
        # Si une cardinalité K est imposée, on fait une pré-optimisation pour trouver les meilleurs actifs
        fixed_indices = []
        if k_cardinality is not None and k_cardinality < n:
            # 1. Optimisation relaxée (sans cardinalité)
            w0 = np.ones(n) / n
            res_relaxed = minimize(objective, w0, method='SLSQP', bounds=bounds, constraints=cons)
            
            if res_relaxed.success:
                # 2. On garde les K plus grands poids
                top_indices = np.argsort(res_relaxed.x)[-k_cardinality:]
                
                # 3. On force les autres à 0 via des bornes
                bounds = []
                for i in range(n):
                    if i in top_indices:
                        bounds.append((0, 1))
                    else:
                        bounds.append((0, 0)) # Force à 0
            else:
                return None # Échec de l'optimisation relaxée

        # Optimisation Finale
        w0 = np.ones(n) / n
        result = minimize(objective, w0, method='SLSQP', bounds=bounds, constraints=cons)
        
        if result.success:
            return result.x
        else:
            return None

    def get_portfolio_metrics(self, w, w_current=None, c_prop=0.0):
        """Retourne les métriques clés pour l'affichage"""
        ret = -self.f1_return(w)
        risk = self.f2_risk(w)
        cost = self.f3_cost(w, w_current, c_prop)
        return ret, risk, cost