import yfinance as yf
import pandas as pd
import streamlit as st
import json
import os

@st.cache_data
def load_tickers_from_json(filename="tick.json"):
    """
    Charge le dictionnaire des secteurs depuis le fichier JSON.
    Gère les chemins relatifs pour éviter les erreurs.
    """
    # Recherche du fichier à plusieurs endroits possibles
    paths_to_check = [
        filename,
        os.path.join(os.path.dirname(os.path.abspath(__file__)), filename),
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), filename)
    ]

    for path in paths_to_check:
        if os.path.exists(path):
            with open(path, 'r') as f:
                return json.load(f)
    
    print(f"Warning: {filename} introuvable.")
    return {}

@st.cache_data(ttl="24h")
def fetch_data(tickers, start_date, end_date):
    """
    Télécharge les données historiques via Yahoo Finance.
    Nettoie les données manquantes et gère les structures multi-index.
    """
    if not tickers:
        return None, None
        
    try:
        # Téléchargement groupé optimisé
        data = yf.download(tickers, start=start_date, end=end_date, group_by='ticker', auto_adjust=True)
        
        if len(tickers) == 1:
            close_data = data['Close'].to_frame()
            close_data.columns = tickers
        else:
            close_data = pd.DataFrame()
            for t in tickers:
                try:
                    # Gestion flexible des colonnes renvoyées par yfinance
                    if (t, 'Close') in data.columns:
                        close_data[t] = data[t]['Close']
                    elif t in data.columns and 'Close' in data[t].columns:
                         close_data[t] = data[t]['Close']
                    elif 'Close' in data.columns and t in data['Close'].columns:
                         close_data[t] = data['Close'][t]
                except:
                    pass
        
        # Nettoyage des données
        close_data = close_data.dropna(axis=1, how='all') # Colonnes vides
        close_data = close_data.fillna(method='ffill').dropna() # Trous
        
        if close_data.empty or close_data.shape[1] < 2:
            return None, None

        # Calculs Financiers (Annualisés)
        returns = close_data.pct_change().dropna()
        mu = returns.mean() * 252
        Sigma = returns.cov() * 252
        
        return mu, Sigma

    except Exception as e:
        print(f"Erreur data: {e}")
        return None, None