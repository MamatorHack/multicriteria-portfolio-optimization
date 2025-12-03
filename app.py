import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from optimizer import PortfolioOptimizer

# Configuration de la page
st.set_page_config(page_title="Optimisation Portefeuille", layout="wide")

# Initialisation du moteur
if 'optimizer' not in st.session_state:
    st.session_state.optimizer = PortfolioOptimizer(n_assets=15)

opt = st.session_state.optimizer

# --- EN-TÊTE ---
st.title("📊 Optimisation de Portefeuille Multi-Critère")
st.markdown("""
Cette application résout le problème d'allocation d'actifs en prenant en compte :
* Le compromis **Rendement / Risque** (Markowitz)
* Les **Coûts de Transaction**
* La **Cardinalité** (Nombre d'actifs max)
""")

# --- SIDEBAR : Paramètres Utilisateur ---
st.sidebar.header("Paramètres de Gestion")

# 1. Paramètres de Coûts
st.sidebar.subheader("1. Contraintes Opérationnelles")
c_prop = st.sidebar.number_input("Coût de transaction (%)", 0.0, 5.0, 0.5, step=0.1) / 100
k_card = st.sidebar.slider("Cardinalité Max (Nb Actifs)", 2, opt.n_assets, 5)

# Portefeuille initial (Simulation: équipondéré)
w_current = np.ones(opt.n_assets) / opt.n_assets

# 2. Paramètre de Rendement Cible
st.sidebar.subheader("2. Objectifs")
r_min_user = st.sidebar.slider("Rendement Annuel Minimal Visé", 0.0, 0.15, 0.05, format="%.2f")

# --- CORPS DE LA PAGE ---

# 1. Calcul de la Frontière Efficiente (Simulation)
st.subheader("1. Frontière Efficiente & Sélection")

# Bouton pour lancer le calcul (peut être lourd)
if st.button("Générer la Frontière de Pareto"):
    with st.spinner("Optimisation en cours..."):
        results = []
        # On fait varier lambda pour tracer la courbe (compromis risque/rendement)
        for lmbda in np.linspace(0, 1, 20):
            w_opt = opt.optimize(w_current=w_current, c_prop=c_prop, k_cardinality=k_card, lmbda=lmbda)
            if w_opt is not None:
                ret, risk, cost = opt.get_portfolio_metrics(w_opt, w_current, c_prop)
                results.append({
                    "Risque (Variance)": risk,
                    "Rendement Espéré": ret,
                    "Coûts": cost,
                    "Poids": w_opt
                })
        
        st.session_state.df_pareto = pd.DataFrame(results)

if 'df_pareto' in st.session_state:
    df = st.session_state.df_pareto
    
    # Graphique Interactif
    fig = px.scatter(
        df, x="Risque (Variance)", y="Rendement Espéré", 
        color="Coûts", size_max=10,
        title="Frontière Efficiente (Niveau 1 & 2)",
        hover_data={"Coûts":':.4f'}
    )
    
    # Ligne de seuil utilisateur [cite: 124]
    fig.add_hline(y=r_min_user, line_dash="dash", line_color="red", annotation_text=f"Min: {r_min_user:.2f}")
    st.plotly_chart(fig, use_container_width=True)

    # --- SÉLECTION OPTIMALE ---
    st.subheader("2. Portefeuille Optimal Sélectionné")
    
    # Filtrer les portefeuilles qui respectent la contrainte r_min
    valid_portfolios = df[df["Rendement Espéré"] >= r_min_user]
    
    if not valid_portfolios.empty:
        # On prend celui avec le risque minimum parmi ceux valides
        best_port = valid_portfolios.sort_values(by="Risque (Variance)").iloc[0]
        
        # Affichage des KPIs
        col1, col2, col3 = st.columns(3)
        col1.metric("Rendement", f"{best_port['Rendement Espéré']:.2%}")
        col2.metric("Risque (Volatilité)", f"{np.sqrt(best_port['Risque (Variance)']):.2%}")
        col3.metric("Coûts de Transaction", f"{best_port['Coûts']:.4f}")
        
        # Analyse de la composition
        w_final = best_port['Poids']
        
        # DataFrame pour l'affichage
        df_alloc = pd.DataFrame({
            "Actif": opt.asset_names,
            "Poids": w_final,
            "Secteur": opt.sectors
        })
        # Filtrer les poids négligeables pour la propreté
        df_alloc = df_alloc[df_alloc["Poids"] > 0.001]
        
        row1_col1, row1_col2 = st.columns(2)
        
        with row1_col1:
            st.markdown("### 🥧 Allocation d'Actifs")
            fig_pie = px.pie(df_alloc, values='Poids', names='Actif', hole=0.4)
            st.plotly_chart(fig_pie, use_container_width=True)
            
        with row1_col2:
            st.markdown("### 🏭 Exposition Sectorielle")
            # Agrégation par secteur [cite: 126]
            df_sector = df_alloc.groupby("Secteur")["Poids"].sum().reset_index()
            fig_bar = px.bar(df_sector, x="Secteur", y="Poids", color="Secteur")
            st.plotly_chart(fig_bar, use_container_width=True)
            
        st.dataframe(df_alloc.style.format({"Poids": "{:.2%}"}))

    else:
        st.warning(f"Impossible de trouver un portefeuille avec un rendement > {r_min_user:.2f} compte tenu des contraintes.")

else:

    st.info("Cliquez sur 'Générer la Frontière de Pareto' pour lancer l'optimisation.")
