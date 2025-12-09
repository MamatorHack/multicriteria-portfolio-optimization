import streamlit as st
import pandas as pd
import numpy as np

# Modules perso
from modules.data_loader import fetch_data, load_tickers_from_json
from modules.optimizer import run_nsga2_optimization, get_best_portfolio
from modules.plots import plot_3d_frontier, plot_sector_pie, plot_projection

# --- CONFIGURATION PAGE ---
st.set_page_config(page_title="Portfolio Manager", layout="wide", page_icon="📈")
st.markdown("<style>.block-container {padding-top: 1rem; padding-bottom: 2rem;}</style>", unsafe_allow_html=True)

# --- NAVIGATION ---
st.sidebar.title("Navigation")
if "nav_selection" not in st.session_state:
    st.session_state["nav_selection"] = "💼 Configuration & Portefeuille"

def go_to_optimization():
    st.session_state.nav_selection = "🚀 Optimisation & Résultats"

page = st.sidebar.radio(
    "Aller vers :", 
    ["💼 Configuration & Portefeuille", "🚀 Optimisation & Résultats"],
    key="nav_selection"
)

# --- CHARGEMENT DONNÉES ---
sectors_dict = load_tickers_from_json("tick.json")

# Initialisation Session State
if 'universe_tickers' not in st.session_state:
    st.session_state['universe_tickers'] = [] 
if 'portfolio_tickers' not in st.session_state:
    st.session_state['portfolio_tickers'] = [] 
if 'portfolio_weights' not in st.session_state:
    st.session_state['portfolio_weights'] = {} 

# =========================================================
# PAGE 1 : CONFIGURATION ET PORTEFEUILLE
# =========================================================
if page == "💼 Configuration & Portefeuille":
    st.title("💼 Configuration de l'Univers")
    
    col_left, col_right = st.columns([1, 1.2], gap="large")
    
    # --- COLONNE GAUCHE : UNIVERS D'INVESTISSEMENT ---
    with col_left:
        st.subheader("1. Univers d'Investissement")
        st.caption("Quels actifs l'algorithme peut-il explorer ?")
        
        if not sectors_dict:
            st.error("⚠️ Fichier 'tick.json' introuvable.")
        else:
            # A. Filtre Secteur
            all_sectors = list(sectors_dict.keys())
            chosen_sectors = st.multiselect("Filtrer par Secteurs :", all_sectors, default=all_sectors[:1])
            
            available_tickers = []
            for sec in chosen_sectors:
                available_tickers.extend(sectors_dict[sec])
            
            st.markdown("---")
            
            # B. Boutons d'action
            c1, c2 = st.columns(2)
            if c1.button("Tout Ajouter (Secteurs)"):
                current = set(st.session_state['universe_tickers'])
                current.update(available_tickers)
                st.session_state['universe_tickers'] = sorted(list(current))
                st.rerun()
            
            if c2.button("Tout Effacer (Univers)"):
                st.session_state['universe_tickers'] = []
                st.rerun()

            # C. Sélection fine
            options_universe = sorted(list(set(available_tickers + st.session_state['universe_tickers'])))
            
            selection_univ = st.multiselect(
                "Sélection manuelle des candidats :", 
                options=options_universe,
                default=st.session_state['universe_tickers']
            )
            
            if selection_univ != st.session_state['universe_tickers']:
                st.session_state['universe_tickers'] = selection_univ
                st.rerun()
                
            st.info(f"⚡ {len(st.session_state['universe_tickers'])} actifs dans le pool d'optimisation.")

    # --- ZONE DROITE : PORTEFEUILLE ACTUEL ---
    with col_right:
        st.subheader("2. Votre Portefeuille Actuel")
        st.caption("Que possédez-vous déjà ? (Recherche globale)")
        
        # 1. Liste complète pour la recherche
        all_tickers_global = []
        for t_list in sectors_dict.values():
            all_tickers_global.extend(t_list)
        all_tickers_global = sorted(list(set(all_tickers_global)))
        
        # 2. Barre de recherche (Hors Formulaire)
        portfolio_selection = st.multiselect(
            "🔍 Rechercher et ajouter vos actifs :",
            options=all_tickers_global,
            default=st.session_state['portfolio_tickers'],
            placeholder="Tapez 'AAPL', 'Total', etc..."
        )
        
        if portfolio_selection != st.session_state['portfolio_tickers']:
            st.session_state['portfolio_tickers'] = portfolio_selection
            st.rerun()

        st.markdown("---")
        
        # 3. Formulaire des poids (Stable)
        if not st.session_state['portfolio_tickers']:
            st.info("Portefeuille vide (100% Cash).")
            st.session_state['portfolio_weights'] = {}
        else:
            st.write("📊 **Répartition du capital (Objectif 100%)**")
            
            with st.form("weights_form"):
                cols = st.columns(3)
                input_values = {}
                
                for i, ticker in enumerate(st.session_state['portfolio_tickers']):
                    col_idx = i % 3
                    with cols[col_idx]:
                        saved_val = st.session_state['portfolio_weights'].get(ticker, 0.0)
                        val = st.number_input(
                            f"{ticker} %",
                            min_value=0.0, max_value=100.0, step=1.0, format="%.1f",
                            value=float(saved_val),
                            key=f"w_input_{ticker}"
                        )
                        input_values[ticker] = val
                
                st.markdown("---")
                submitted = st.form_submit_button("💾 VALIDER LA RÉPARTITION", type="primary")
            
            if submitted:
                total_w = sum(input_values.values())
                st.session_state['portfolio_weights'] = input_values
                
                if abs(total_w - 100) < 0.1:
                    st.success(f"✅ Parfait ! Total : {total_w:.1f}%")
                elif total_w == 0:
                    st.warning("⚠️ Tous les poids sont à 0%. Considéré comme 100% Cash.")
                else:
                    st.warning(f"ℹ️ Total : {total_w:.1f}% (Reste {100-total_w:.1f}% en Cash)")

        # Paramètres Globaux
        st.subheader("3. Paramètres")
        c1, c2 = st.columns(2)
        st.session_state['start_date'] = c1.date_input("Début", pd.to_datetime("2021-01-01"))
        st.session_state['end_date'] = c2.date_input("Fin", pd.to_datetime("2023-12-31"))
        st.session_state['capital'] = st.number_input("Capital ($)", 1000, 10000000, 10000, step=1000)

    # Navigation
    st.markdown("---")
    st.button("Lancer l'Optimisation ➡️", type="primary", on_click=go_to_optimization)

# =========================================================
# PAGE 2 : OPTIMISATION
# =========================================================
elif page == "🚀 Optimisation & Résultats":
    st.title("🚀 Optimisation Multi-Critères (NSGA-II)")
    
    # Fusion des listes
    final_tickers = list(set(st.session_state['universe_tickers'] + st.session_state['portfolio_tickers']))
    
    if not final_tickers:
        st.warning("⚠️ Aucun actif sélectionné.")
        st.stop()
        
    # Sidebar
    st.sidebar.header("Contraintes")
    nb_assets = len(final_tickers)
    max_k = min(nb_assets, 20)
    K = st.sidebar.slider("Cardinalité Cible", 2, nb_assets, min(5, max_k))
    c_prop = st.sidebar.number_input("Frais Transaction (%)", 0.0, 5.0, 0.5, step=0.1) / 100

    st.write(f"Analyse sur **{len(final_tickers)}** actifs.")

    # Chargement
    with st.spinner("Récupération des données..."):
        mu, Sigma = fetch_data(
            final_tickers, 
            st.session_state['start_date'], 
            st.session_state['end_date']
        )
        
    if mu is None:
        st.error("Erreur de données.")
        st.stop()

    # Alignement des poids
    valid_tickers = mu.index.tolist()
    w_initial_aligned = []
    
    for t in valid_tickers:
        val = st.session_state['portfolio_weights'].get(t, 0.0)
        w_initial_aligned.append(val)
    
    w_initial = np.array(w_initial_aligned)
    if np.sum(w_initial) > 0:
        w_initial = w_initial / 100.0 # Décimal
        
    # Simulation
    if st.button("⚡ LANCER LE CALCUL (NSGA-II)", type="primary"):
        with st.spinner("Optimisation Génétique en cours..."):
            # CORRECTION ICI : On a retiré le 'None' qui causait le crash
            df_results = run_nsga2_optimization(
                mu.values, 
                Sigma.values, 
                w_initial, 
                K, 
                c_prop, 
                valid_tickers
            )
            st.session_state['df_results'] = df_results

    # Résultats
    if 'df_results' in st.session_state:
        df = st.session_state['df_results']
        capital = st.session_state['capital']
        
        st.divider()
        c_main, c_side = st.columns([3, 1])
        
        with c_main:
            min_ret = st.slider("🎯 Rendement Visé", 
                                float(df['Rendement'].min()), float(df['Rendement'].max()), float(df['Rendement'].mean()))
            best = get_best_portfolio(df, min_ret)
            
            if best is not None:
                st.plotly_chart(plot_3d_frontier(df, best, w_initial, mu.values, Sigma.values, capital), use_container_width=True)
                st.plotly_chart(plot_projection(capital, best['Rendement'], best['Risque']), use_container_width=True)
            else:
                st.warning("Inatteignable.")

        with c_side:
            if best is not None:
                st.success("### Optimum")
                st.metric("Rendement", f"{best['Rendement']:.2%}")
                st.metric("Risque", f"{best['Risque']:.2%}")
                st.metric("Coûts", f"{best['Coût']*capital:.2f} $")
                
                active_map = {}
                for sec, t_list in sectors_dict.items():
                    for t in t_list:
                        if t in valid_tickers: active_map[t] = sec
                
                st.plotly_chart(plot_sector_pie(best, valid_tickers, active_map), use_container_width=True)

        if best is not None:
            st.divider()
            st.subheader("📋 Plan d'Arbitrage")
            plan = []
            buy_t, sell_t = 0, 0
            for i, t in enumerate(valid_tickers):
                tgt = best[t]
                curr = w_initial[i]
                diff = (tgt - curr) * capital
                
                if abs(diff) > 5 or (tgt > 0 and curr > 0):
                    op = "ACHAT 🟢" if diff > 0 else "VENTE 🔴"
                    if abs(diff) < 5: op = "CONSERVER 🔵"
                    if diff > 0: buy_t += diff
                    else: sell_t += abs(diff)
                    plan.append({"Actif": t, "Secteur": active_map.get(t, "N/A"), "Op": op, "Actuel": f"{curr:.1%}", "Cible": f"{tgt:.1%}", "Flux": f"{diff:+.2f}$"})
            
            st.dataframe(pd.DataFrame(plan), use_container_width=True)
            c1, c2 = st.columns(2)
            c1.info(f"Ventes: {sell_t:.2f}$")
            c2.success(f"Achats: {buy_t:.2f}$")