import numpy as np
import pandas as pd
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.optimize import minimize
from pymoo.termination import get_termination

class PortfolioProblem(ElementwiseProblem):
    def __init__(self, mu, Sigma, w_init, k, cost_rate, tickers):
        # 3 Objectifs à minimiser : Risque, -Rendement, Coût
        super().__init__(n_var=len(mu), n_obj=3, n_ieq_constr=0, xl=0.0, xu=1.0)
        self.mu = mu
        self.Sigma = Sigma
        self.w_init = w_init
        self.k = k
        self.cost_rate = cost_rate
        self.tickers = tickers

    def _evaluate(self, x, out, *args, **kwargs):
        # --- 1. RÉPARATION (Gestion Cardinalité K) ---
        # L'algo génétique propose des poids bruts 'x'.
        # On ne garde que les K plus grands pour forcer la cardinalité.
        
        # On identifie les indices des K plus grandes valeurs
        idx_sorted = np.argsort(x)
        idx_keep = idx_sorted[-self.k:] 
        
        # On crée le vecteur propre
        w = np.zeros(len(x))
        w[idx_keep] = x[idx_keep]
        
        # --- 2. NORMALISATION (Budget = 100%) ---
        total_w = np.sum(w)
        if total_w > 1e-6:
            w = w / total_w
        else:
            # Fallback si tout est à 0 (rare)
            w[idx_keep[0]] = 1.0
            
        # --- 3. CALCUL DES OBJECTIFS ---
        # f1: Risque (Volatilité) -> Minimiser
        vol = np.sqrt(np.dot(w.T, np.dot(self.Sigma, w)))
        
        # f2: Rendement -> Maximiser (donc on minimise l'opposé)
        ret = np.sum(self.mu * w)
        neg_ret = -ret
        
        # f3: Coûts -> Minimiser
        turnover = np.sum(np.abs(w - self.w_init))
        cost = turnover * self.cost_rate
        
        # Sortie pour pymoo
        out["F"] = [vol, neg_ret, cost]
        
        # On stocke les poids nettoyés pour pouvoir les récupérer plus tard
        # (Astuce: on les attache à l'objet out pour post-traitement, 
        # bien que pymoo ne l'utilise pas directement)
        out["hash"] = w 

def run_nsga2_optimization(mu, Sigma, w_init, K, cost_rate, ticker_names):
    """
    Lance l'optimisation NSGA-II.
    Remplace l'ancienne simulation Monte Carlo.
    """
    # Configuration du problème
    problem = PortfolioProblem(mu, Sigma, w_init, K, cost_rate, ticker_names)

    # Configuration de l'algorithme génétique
    algorithm = NSGA2(
        pop_size=100,                # Taille population (100 portefeuilles qui évoluent)
        n_offsprings=50,             # Nouveaux enfants par génération
        sampling=FloatRandomSampling(),
        crossover=SBX(prob=0.9, eta=15), # Croisement
        mutation=PM(prob=0.1, eta=20),   # Mutation
        eliminate_duplicates=True
    )

    # Critère d'arrêt : 50 générations (suffisant pour une app web réactive)
    termination = get_termination("n_gen", 50)

    # Exécution
    res = minimize(problem,
                   algorithm,
                   termination,
                   seed=1,
                   verbose=False)

    # --- RECONSTRUCTION DES RÉSULTATS ---
    # NSGA-II renvoie les inputs 'X' optimaux, mais on doit réappliquer la logique
    # de réparation (K actifs + Normalisation) pour avoir les vrais poids finaux.
    
    final_results = []
    
    for i, raw_x in enumerate(res.X):
        # Réapplication de la logique de nettoyage (Cardinalité + Norm)
        idx_sorted = np.argsort(raw_x)
        idx_keep = idx_sorted[-K:]
        w = np.zeros(len(raw_x))
        w[idx_keep] = raw_x[idx_keep]
        w /= np.sum(w)
        
        # Récupération des objectifs finaux
        # res.F contient [Vol, -Ret, Cost]
        vol = res.F[i][0]
        ret = -res.F[i][1] # On remet en positif
        cost = res.F[i][2]
        
        # Création du dictionnaire
        row = {
            'Risque': vol,
            'Rendement': ret,
            'Coût': cost,
            'Sharpe': (ret / vol) if vol > 0 else 0
        }
        
        # Ajout des poids détaillés
        for idx_asset, ticker in enumerate(ticker_names):
            row[ticker] = w[idx_asset]
            
        final_results.append(row)

    # On ajoute quelques points aléatoires pour "remplir" un peu le graphique 3D
    # car NSGA-II est parfois "trop" efficace et ne trouve qu'une ligne fine.
    # (Optionnel, mais joli pour la visualisation)
    return pd.DataFrame(final_results)

def get_best_portfolio(df_results, min_return):
    """
    Trouve le portefeuille optimal : celui qui minimise le Risque 
    pour un niveau de Rendement donné.
    """
    # Filtre sur le rendement minimum exigé
    df_filtered = df_results[df_results['Rendement'] >= min_return]
    
    if df_filtered.empty:
        return None
    
    # Retourne la ligne avec le risque minimal
    return df_filtered.loc[df_filtered['Risque'].idxmin()]