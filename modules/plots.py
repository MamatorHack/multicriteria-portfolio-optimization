import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

def plot_3d_frontier(df, best_port, w_initial, mu, Sigma, capital):
    """Génère le graphique 3D interactif (Rendement / Risque / Coût)."""
    
    fig = px.scatter_3d(
        df, x='Risque', y='Rendement', z='Coût',
        color='Sharpe', opacity=0.3,
        color_continuous_scale='Viridis',
        title="Nuage de Portefeuilles (Frontière Efficiente)",
        height=650
    )
    
    # Marqueur pour le portefeuille optimal
    if best_port is not None:
        fig.add_trace(go.Scatter3d(
            x=[best_port['Risque']], y=[best_port['Rendement']], z=[best_port['Coût']],
            mode='markers', marker=dict(size=20, color='red', symbol='diamond'),
            name='OPTIMUM RECOMMANDÉ'
        ))
    
    # Marqueur pour le portefeuille initial (si existant)
    if mu is not None and Sigma is not None:
        init_ret = np.sum(mu * w_initial)
        init_vol = np.sqrt(np.dot(w_initial.T, np.dot(Sigma, w_initial)))
        fig.add_trace(go.Scatter3d(
            x=[init_vol], y=[init_ret], z=[0],
            mode='markers', marker=dict(size=15, color='blue', symbol='circle'),
            name='Votre Portefeuille'
        ))
        
    fig.update_layout(scene=dict(
        xaxis_title='Risque (Volatilité)', 
        yaxis_title='Rendement Espéré', 
        zaxis_title='Coûts Transaction'
    ))
    return fig

def plot_sector_pie(best_port, tickers, sectors):
    """Génère le camembert de répartition sectorielle."""
    sector_alloc = {}
    for t in tickers:
        if t in best_port:
            sec = sectors.get(t, "Autre")
            sector_alloc[sec] = sector_alloc.get(sec, 0) + best_port[t]
    
    df_sec = pd.DataFrame(list(sector_alloc.items()), columns=['Sec', 'Poids'])
    fig = px.pie(df_sec, values='Poids', names='Sec', hole=0.4, title="Exposition Sectorielle")
    fig.update_layout(showlegend=True, height=300, margin=dict(t=30, b=0, l=0, r=0))
    return fig

def plot_projection(capital, r_ann, sig_ann, years=10):
    """Génère le graphique de projection de fortune (Intervalle de confiance)."""
    time_range = list(range(years + 1))
    
    # Modèle simpliste : Rendement +/- 1 Ecart-type
    balance_optimistic = [capital * ((1 + r_ann + sig_ann) ** y) for y in time_range]
    balance_expected = [capital * ((1 + r_ann) ** y) for y in time_range]
    balance_pessimistic = [capital * ((1 + r_ann - sig_ann) ** y) for y in time_range]
    
    fig = go.Figure()
    
    # Zone d'incertitude
    fig.add_trace(go.Scatter(x=time_range, y=balance_pessimistic, mode='lines', line=dict(width=0), showlegend=False))
    fig.add_trace(go.Scatter(
        x=time_range, y=balance_optimistic, mode='lines', 
        line=dict(width=0), fill='tonexty', fillcolor='rgba(0, 200, 100, 0.2)',
        name='Intervalle de Confiance'
    ))
    
    # Scénario central
    fig.add_trace(go.Scatter(
        x=time_range, y=balance_expected, mode='lines+markers',
        line=dict(color='green', width=4), name='Scénario Moyen'
    ))
    
    fig.update_layout(
        title="Projection de Fortune (10 ans)",
        xaxis_title="Années", yaxis_title="Capital ($)",
        hovermode="x unified", height=400
    )
    return fig