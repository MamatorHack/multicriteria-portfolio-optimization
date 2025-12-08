import streamlit as st
import pandas as pd
import numpy as np

# Importation des modules locaux
from modules.data_loader import fetch_data, load_tickers_from_json
from modules.optimizer import run_nsga2_optimization, get_best_portfolio
from modules.plots import plot_3d_frontier, plot_sector_pie, plot_projection

# --- 1. CONFIGURATION GLOBALE ---
st.set_page_config(page_title="Portfolio Manager", layout="wide", page_icon="📈")
st.markdown("<style>.block-container {padding-top: 1rem; padding-bottom: 2rem;}</style>", unsafe_allow_html=True)

# --- 2. GESTION DE LA NAVIGATION (SESSION STATE) ---
st.sidebar.title("Navigation")

if "nav_selection" not in st.session_state:
    st.session_state["nav_selection"] = "💼 Configuration & Portefeuille"

def go_to_optimization():
    """Callback pour changer de page via un bouton."""
    st.session_state.nav_selection = "🚀 Optimisation & Résultats"

# Widget de navigation
page = st.sidebar.radio(
    "Aller vers :", 
    ["💼 Configuration & Portefeuille", "🚀 Optimisation & Résultats"],
    key="nav_selection"
)

# --- 3. INITIALISATION ---
# Chargement de la liste des actifs
sectors_dict = load_tickers_from_json("tick.json")

# Initialisation des variables de session
if 'selected_tickers' not in st.session_state:
    st.session_state['selected_tickers'] = []
if 'portfolio_weights' not in st.session_state:
    st.session_state['portfolio_weights'] = {} 

# =========================================================
# PAGE 1 : CONFIGURATION ET PORTEFEUILLE
# =========================================================
if page == "💼 Configuration & Portefeuille":
    st.title("💼 Configuration de l'Univers")
    
    col1, col2 = st.columns([1, 1.2], gap="large")
    
    # --- COLONNE GAUCHE : SÉLECTION DES ACTIFS ---
    with col1:
        st.subheader("1. Univers d'Investissement")
        
        if not sectors_dict:
            st.error("⚠️ Erreur critique : Fichier 'tick.json' introuvable.")
        else:
            # A. Filtre par Secteurs
            all_sectors = list(sectors_dict.keys())
            chosen_sectors = st.multiselect(
                "Filtrer par Secteurs :", 
                all_sectors, 
                default=all_sectors[:2]
            )
            
            # Récupération des tickers disponibles
            available_tickers = []
            for sec in chosen_sectors:
                available_tickers.extend(sectors_dict[sec])
            
            # B. Boutons d'Action Rapide
            c_btn1, c_btn2 = st.columns(2)
            if c_btn1.button("➕ Ajouter la liste visible"):
                current = set(st.session_state['selected_tickers'])
                current.update(available_tickers)
                st.session_state['selected_tickers'] = list(current)
                st.rerun()
            
            if c_btn2.button("🗑️ Réinitialiser tout"):
                st.session_state['selected_tickers'] = []
                st.session_state['portfolio_weights'] = {} 
                st.rerun()

            # C. Sélection Fine
            options = sorted(list(set(available_tickers + st.session_state['selected_tickers'])))
            
            selection = st.multiselect(
                "Sélectionnez vos actifs :", 
                options=options,
                default=st.session_state['selected_tickers']
            )
            
            # Mise à jour si changement manuel
            if selection != st.session_state['selected_tickers']:
                st.session_state['selected_tickers'] = selection
                st.rerun()

    # --- COLONNE DROITE : PARAMÈTRES ET ALLOCATION ---
    with col2:
        st.subheader("2. Paramètres & Allocation")
        
        # Paramètres Globaux
        c_p1, c_p2 = st.columns(2)
        start_date = c_p1.date_input("Début Analyse", pd.to_datetime("2021-01-01"))
        end_date = c_p2.date_input("Fin Analyse", pd.to_datetime("2023-12-31"))
        capital = st.number_input("Capital Total ($)", 1000, 10000000, 10000, step=1000)
        
        # Sauvegarde immédiate
        st.session_state['start_date'] = start_date
        st.session_state['end_date'] = end_date
        st.session_state['capital'] = capital

        st.divider()
        
        # --- FORMULAIRE D'ALLOCATION (Stable & Robuste) ---
        st.info("👇 Modifiez les poids ci-dessous. Cliquez sur 'Valider' pour sauvegarder.")
        
        if not st.session_state['selected_tickers']:
            st.warning("👈 Sélectionnez d'abord des actifs à gauche.")
        else:
            with st.form("allocation_form"):
                cols = st.columns(4) # Grille de 4 colonnes
                
                # Génération dynamique des inputs
                for i, ticker in enumerate(st.session_state['selected_tickers']):
                    col_idx = i % 4
                    with cols[col_idx]:
                        saved_val = st.session_state['portfolio_weights'].get(ticker, 0.0)
                        st.number_input(
                            f"{ticker}",
                            min_value=0.0, max_value=100.0, step=1.0, format="%.1f",
                            value=float(saved_val),
                            key=f"form_input_{ticker}" # Clé utilisée pour la récupération
                        )
                
                st.markdown("---")
                submitted = st.form_submit_button("💾 VALIDER ET SAUVEGARDER", type="primary")
            
            # Traitement après validation
            if submitted:
                total_w = 0
                new_weights = {}
                
                for ticker in st.session_state['selected_tickers']:
                    val = st.session_state.get(f"form_input_{ticker}", 0.0)
                    if val > 0:
                        new_weights[ticker] = val
                        total_w += val
                
                st.session_state['portfolio_weights'] = new_weights
                
                # Feedback utilisateur
                if abs(total_w - 100) < 0.1:
                    st.success(f"✅ Allocation validée ! Total : {total_w:.1f}%")
                elif total_w == 0:
                    st.warning("ℹ️ Portefeuille validé (100% Cash).")
                else:
                    st.warning(f"⚠️ Allocation incomplète : {total_w:.1f}% investis.")
            
            # Bouton de Navigation
            st.markdown(" ")
            col_void, col_next = st.columns([3, 2])
            with col_next:
                st.button("Passer à l'Optimisation ➡️", type="secondary", on_click=go_to_optimization)

# =========================================================
# PAGE 2 : OPTIMISATION & RÉSULTATS
# =========================================================
elif page == "🚀 Optimisation & Résultats":
    st.title("🚀 Optimisation Multi-Critères")
    
    # Vérification de sécurité
    if not st.session_state['selected_tickers']:
        st.warning("⚠️ Aucun actif sélectionné. Veuillez retourner à la configuration.")
        st.stop()
        
    # --- Sidebar : Contraintes spécifiques ---
    st.sidebar.header("Contraintes")
    nb_assets = len(st.session_state['selected_tickers'])
    max_k = min(nb_assets, 15)
    K = st.sidebar.slider("Cardinalité Cible (Nb Actifs Max)", 2, nb_assets, min(5, max_k))
    c_prop = st.sidebar.number_input("Frais Transaction (%)", 0.0, 5.0, 0.5, step=0.1) / 100

    # --- Chargement des Données ---
    with st.spinner("Récupération des données financières (Yahoo Finance)..."):
        mu, Sigma = fetch_data(
            st.session_state['selected_tickers'], 
            st.session_state['start_date'], 
            st.session_state['end_date']
        )
        
    if mu is None:
        st.error("❌ Erreur de téléchargement des données. Vérifiez votre connexion internet.")
        st.stop()

    # --- Alignement des Poids (Correctif Dimension) ---
    # On ne garde que les actifs qui ont été correctement téléchargés
    valid_tickers = mu.index.tolist()
    w_initial_aligned = []
    
    for t in valid_tickers:
        val = st.session_state['portfolio_weights'].get(t, 0.0)
        w_initial_aligned.append(val)
    
    w_initial = np.array(w_initial_aligned)
    if np.sum(w_initial) > 0:
        w_initial = w_initial / 100.0 # Normalisation
        
    if len(w_initial) != len(mu):
        st.error(f"Erreur de dimension : {len(w_initial)} vs {len(mu)}")
        st.stop()

    # --- Bouton de Simulation ---
    if st.button("⚡ LANCER L'OPTIMISATION (NSGA-II)", type="primary"):
        with st.spinner(f"Évolution Génétique en cours sur {len(valid_tickers)} actifs..."):
            
            # Appel à la nouvelle fonction NSGA-II (via l'alias ou direct)
            # Note : on n'a plus besoin de n_sims
            df_results = run_nsga2_optimization(
                mu.values, 
                Sigma.values, 
                w_initial, 
                K, 
                c_prop, 
                valid_tickers
            )
            st.session_state['df_results'] = df_results
            
            
    # --- Affichage des Résultats ---
    if 'df_results' in st.session_state:
        df = st.session_state['df_results']
        capital = st.session_state['capital']
        
        st.divider()
        c_main, c_side = st.columns([3, 1])
        
        # Slider de sélection
        with c_main:
            min_ret = st.slider("🎯 Rendement Annuel Visé", 
                                float(df['Rendement'].min()), 
                                float(df['Rendement'].max()), 
                                float(df['Rendement'].mean()))
            
            best = get_best_portfolio(df, min_ret)
            
            if best is not None:
                st.plotly_chart(plot_3d_frontier(df, best, w_initial, mu.values, Sigma.values, capital), use_container_width=True)
                st.plotly_chart(plot_projection(capital, best['Rendement'], best['Risque']), use_container_width=True)
            else:
                st.warning("Aucun portefeuille ne correspond à ce critère.")

        # KPI et Camembert
        with c_side:
            if best is not None:
                st.success("### Optimum Trouvé")
                st.metric("Rendement", f"{best['Rendement']:.2%}")
                st.metric("Risque", f"{best['Risque']:.2%}")
                st.metric("Coûts", f"{best['Coût']*capital:.2f} $")
                
                # Mapping pour le camembert
                active_map = {}
                for sec, t_list in sectors_dict.items():
                    for t in t_list:
                        if t in valid_tickers: active_map[t] = sec
                
                st.plotly_chart(plot_sector_pie(best, valid_tickers, active_map), use_container_width=True)

        # Tableau d'Arbitrage
        if best is not None:
            st.divider()
            st.subheader("📋 Plan d'Arbitrage Recommandé")
            plan = []
            buy_t, sell_t = 0, 0
            
            for i, t in enumerate(valid_tickers):
                tgt = best[t]
                curr = w_initial[i]
                diff = (tgt - curr) * capital
                
                # On filtre les mouvements insignifiants
                if abs(diff) > 5 or (tgt > 0 and curr > 0):
                    op = "ACHAT 🟢" if diff > 0 else "VENTE 🔴"
                    if abs(diff) < 5: op = "CONSERVER 🔵"
                    
                    if diff > 0: buy_t += diff
                    else: sell_t += abs(diff)
                    
                    plan.append({
                        "Actif": t, 
                        "Secteur": active_map.get(t, "N/A"), 
                        "Opération": op, 
                        "Poids Actuel": f"{curr:.1%}", 
                        "Poids Cible": f"{tgt:.1%}", 
                        "Volume ($)": f"{diff:+.2f} $"
                    })
            
            st.dataframe(pd.DataFrame(plan), use_container_width=True)
            
            c1, c2 = st.columns(2)
            c1.info(f"💰 Ventes (Cash In) : {sell_t:.2f} $")
            c2.success(f"💸 Achats (Cash Out) : {buy_t:.2f} $")