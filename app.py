import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os

# Importation de la classe d'optimisation
# Assurez-vous que le fichier optimizer.py est bien dans le même dossier
from optimizer import PortfolioOptimizer

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(
    page_title="Optimisation de Portefeuille Réelle",
    page_icon="📈",
    layout="wide"
)

# --- INITIALISATION DU MOTEUR (CACHE) ---
if 'optimizer' not in st.session_state:
    try:
        # On tente d'initialiser l'optimiseur qui va lire le dossier 'data'
        st.session_state.optimizer = PortfolioOptimizer(data_folder="data")
    except FileNotFoundError:
        st.error("⚠️ Erreur critique : Le dossier 'data' est introuvable ou vide.")
        st.info("Veuillez d'abord exécuter le script 'download.py' pour récupérer les données historiques.")
        st.stop()
    except Exception as e:
        st.error(f"Une erreur est survenue lors du chargement des données : {e}")
        st.stop()

opt = st.session_state.optimizer

# --- EN-TÊTE ---
st.title("📈 Optimisation de Portefeuille (Données Réelles)")
st.markdown(f"""
Cette application utilise les données historiques de **{opt.n_assets} actifs** (téléchargés via Yahoo Finance) pour construire un portefeuille optimal.
Elle prend en compte le rendement, le risque (volatilité) et les coûts de transaction.
""")

# --- SIDEBAR : PARAMÈTRES ---
st.sidebar.header("⚙️ Paramètres de Gestion")

# 1. Infos sur l'univers
st.sidebar.info(f"✅ **{opt.n_assets} actifs** chargés depuis la base de données.")

# 2. Coûts et Contraintes
st.sidebar.subheader("1. Contraintes Opérationnelles")
c_prop_input = st.sidebar.number_input(
    "Coût de transaction (%)", 
    min_value=0.0, max_value=5.0, value=0.5, step=0.1
)
c_prop = c_prop_input / 100.0  # Conversion en décimal

# Cardinalité (ne peut pas dépasser le nombre total d'actifs)
k_card = st.sidebar.slider(
    "Cardinalité Max (Nb Actifs en portefeuille)", 
    min_value=2, max_value=opt.n_assets, value=min(10, opt.n_assets)
)

# 3. Objectifs
st.sidebar.subheader("2. Objectifs de Performance")
# Plage ajustée pour des données réelles (de -10% à +40% par an)
r_min_user = st.sidebar.slider(
    "Rendement Annuel Minimal Visé", 
    min_value=-0.10, max_value=0.40, value=0.10, step=0.01, format="%.2f"
)

# Portefeuille initial (Simulation: on suppose qu'on part de liquidités, donc w_current=None ou équilibré)
# Pour l'exercice, on part d'un portefeuille vide (tout en cash) ou équipondéré.
# Ici, on prend un portefeuille équipondéré comme référence pour le calcul des coûts de transaction.
w_current = np.ones(opt.n_assets) / opt.n_assets

# --- CORPS DE LA PAGE ---

st.subheader("1. Frontière Efficiente & Analyse")

col_action, col_info = st.columns([1, 3])

with col_action:
    generate_btn = st.button("🚀 Calculer la Frontière de Pareto", use_container_width=True)

# Logique de calcul
if generate_btn or 'df_pareto' in st.session_state:
    
    if generate_btn:
        with st.spinner("Optimisation mathématique en cours sur données réelles..."):
            results = []
            # On balaye lambda de 0 (Rendement max) à 1 (Risque min)
            # On augmente le pas pour avoir une belle courbe
            lambdas = np.linspace(0, 1, 25) 
            
            progress_bar = st.progress(0)
            
            for i, lmbda in enumerate(lambdas):
                w_opt = opt.optimize(
                    w_current=w_current, 
                    c_prop=c_prop, 
                    k_cardinality=k_card, 
                    lmbda=lmbda
                )
                
                if w_opt is not None:
                    ret, risk, cost = opt.get_portfolio_metrics(w_opt, w_current, c_prop)
                    # On stocke les résultats
                    results.append({
                        "Risque (Variance)": risk,
                        "Volatilité (Ecart-Type)": np.sqrt(risk),
                        "Rendement Espéré": ret,
                        "Coûts Transaction": cost,
                        "Poids": w_opt,
                        "Lambda": lmbda
                    })
                
                progress_bar.progress((i + 1) / len(lambdas))
            
            st.session_state.df_pareto = pd.DataFrame(results)
            progress_bar.empty()

    # Affichage si les données existent
    if 'df_pareto' in st.session_state and not st.session_state.df_pareto.empty:
        df = st.session_state.df_pareto
        
        # --- Graphique de la Frontière ---
        fig = px.scatter(
            df, 
            x="Volatilité (Ecart-Type)", 
            y="Rendement Espéré", 
            color="Coûts Transaction",
            size="Coûts Transaction", # La taille des bulles dépend du coût
            title="Frontière Efficiente (Compromis Risque / Rendement)",
            labels={"Volatilité (Ecart-Type)": "Risque (Volatilité Annuelle)", "Rendement Espéré": "Rendement Espéré (Annuel)"},
            hover_data={"Lambda":':.2f'}
        )
        
        # Ligne du rendement minimal visé
        fig.add_hline(y=r_min_user, line_dash="dash", line_color="red", annotation_text=f"Objectif Min: {r_min_user:.1%}")
        st.plotly_chart(fig, use_container_width=True)
        
        # --- SÉLECTION DU MEILLEUR PORTEFEUILLE ---
        st.divider()
        st.subheader("2. Allocation Optimale Sélectionnée")
        
        # Filtrage : On garde uniquement ceux qui satisfont r_min
        valid_portfolios = df[df["Rendement Espéré"] >= r_min_user]
        
        if not valid_portfolios.empty:
            # Critère de choix : Minimiser le Risque parmi ceux qui sont valides
            best_port = valid_portfolios.sort_values(by="Volatilité (Ecart-Type)").iloc[0]
            
            # Affichage des KPIs
            kpi1, kpi2, kpi3 = st.columns(3)
            kpi1.metric("Rendement Annuel", f"{best_port['Rendement Espéré']:.2%}")
            kpi2.metric("Volatilité (Risque)", f"{best_port['Volatilité (Ecart-Type)']:.2%}")
            kpi3.metric("Coûts de Réallocation", f"{best_port['Coûts Transaction']:.4f}")
            
            # --- Analyse de la Composition ---
            w_final = best_port['Poids']
            
            df_alloc = pd.DataFrame({
                "Ticker": opt.asset_names,
                "Poids": w_final,
                "Secteur": opt.sectors
            })
            
            # On filtre les poids négligeables (< 1%) pour la lisibilité
            df_alloc_filtered = df_alloc[df_alloc["Poids"] > 0.01].sort_values(by="Poids", ascending=False)
            
            col_pie, col_bar = st.columns(2)
            
            with col_pie:
                st.markdown("#### 🥧 Répartition par Actif")
                if not df_alloc_filtered.empty:
                    fig_pie = px.pie(df_alloc_filtered, values='Poids', names='Ticker', hole=0.4)
                    st.plotly_chart(fig_pie, use_container_width=True)
                else:
                    st.info("Aucun actif n'a un poids significatif (>1%).")
            
            with col_bar:
                st.markdown("#### 🏭 Exposition Sectorielle")
                # Agrégation par secteur
                df_sector = df_alloc.groupby("Secteur")["Poids"].sum().reset_index().sort_values(by="Poids", ascending=False)
                fig_bar = px.bar(
                    df_sector, x="Secteur", y="Poids", 
                    color="Secteur", text_auto='.1%'
                )
                fig_bar.update_layout(showlegend=False)
                st.plotly_chart(fig_bar, use_container_width=True)
            
            # Tableau détaillé
            with st.expander("Voir le détail complet des poids"):
                st.dataframe(df_alloc.sort_values(by="Poids", ascending=False).style.format({"Poids": "{:.2%}"}))
                
        else:
            st.warning(f"❌ Aucun portefeuille trouvé avec un rendement supérieur à {r_min_user:.1%}. Essayez de réduire votre objectif de rendement ou d'augmenter la tolérance au risque.")
    
    elif generate_btn:
        st.warning("L'optimisation n'a pas trouvé de solutions convergentes. Vérifiez les contraintes.")

else:
    st.info("👋 Cliquez sur le bouton 'Calculer' ci-dessus pour lancer l'optimisation.")
