"""
Module de visualisation des données et résultats
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))
from config.config import *

# Configuration matplotlib
plt.style.use('default')
sns.set_palette("husl")

class DataVisualizer:
    def __init__(self):
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        
    def plot_dropout_distribution(self, df):
        """Graphique de distribution de l'abandon scolaire"""
        dropout_counts = df['dropout'].value_counts()
        
        # Conversion des valeurs si nécessaire
        if 0 in dropout_counts.index and 1 in dropout_counts.index:
            dropout_counts.index = ['Non-Abandon', 'Abandon']
        elif 'No' in dropout_counts.index and 'Yes' in dropout_counts.index:
            dropout_counts.index = ['Non-Abandon', 'Abandon']
        
        fig = px.pie(
            values=dropout_counts.values,
            names=dropout_counts.index,
            title="Distribution de l'Abandon Scolaire",
            color_discrete_sequence=self.colors
        )
        
        return fig
    
    def plot_feature_distributions(self, df):
        """Histogrammes des variables principales"""
        numeric_cols = ['age', 'average_grade', 'absenteeism_rate', 'satisfaction_score']
        existing_cols = [col for col in numeric_cols if col in df.columns]
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=existing_cols,
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        for i, col in enumerate(existing_cols):
            row = (i // 2) + 1
            col_pos = (i % 2) + 1
            
            fig.add_trace(
                go.Histogram(x=df[col], name=col, showlegend=False),
                row=row, col=col_pos
            )
        
        fig.update_layout(
            title="Distribution des Variables Principales",
            height=600
        )
        
        return fig
    
    def plot_correlation_heatmap(self, df):
        """Heatmap de corrélation"""
        # Sélection des colonnes numériques
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col != 'student_id']
        
        corr_matrix = df[numeric_cols].corr()
        
        fig = px.imshow(
            corr_matrix,
            title="Matrice de Corrélation",
            color_continuous_scale="RdBu",
            aspect="auto"
        )
        
        return fig
    
    def plot_dropout_by_category(self, df, category_col):
        """Graphique d'abandon par catégorie"""
        if category_col not in df.columns:
            return None
        
        # Calcul des taux d'abandon par catégorie
        dropout_by_cat = df.groupby(category_col)['dropout'].agg(['count', 'sum']).reset_index()
        dropout_by_cat['dropout_rate'] = dropout_by_cat['sum'] / dropout_by_cat['count']
        
        fig = px.bar(
            dropout_by_cat,
            x=category_col,
            y='dropout_rate',
            title=f"Taux d'Abandon par {category_col}",
            labels={'dropout_rate': 'Taux d\'Abandon'}
        )
        
        return fig
    
    def plot_cluster_visualization(self, df_clustered, pca_data=None):
        """Visualisation des clusters"""
        if 'cluster' not in df_clustered.columns:
            return None
        
        if pca_data is not None:
            # Visualisation 2D avec PCA
            fig = px.scatter(
                x=pca_data[:, 0],
                y=pca_data[:, 1],
                color=df_clustered['cluster'].astype(str),
                title="Visualisation des Clusters (PCA)",
                labels={'x': 'Composante Principale 1', 'y': 'Composante Principale 2'}
            )
        else:
            # Utilisation de deux variables principales
            fig = px.scatter(
                df_clustered,
                x='average_grade',
                y='satisfaction_score',
                color='cluster',
                title="Visualisation des Clusters",
                size='absenteeism_rate' if 'absenteeism_rate' in df_clustered.columns else None
            )
        
        return fig
    
    def plot_feature_importance(self, importance_df):
        """Graphique d'importance des variables"""
        if importance_df is None or importance_df.empty:
            return None
        
        # Limitation aux 10 variables les plus importantes
        top_features = importance_df.head(10)
        
        fig = px.bar(
            top_features,
            x='importance',
            y='feature',
            orientation='h',
            title="Importance des Variables",
            labels={'importance': 'Importance', 'feature': 'Variables'}
        )
        
        fig.update_layout(yaxis={'categoryorder': 'total ascending'})
        
        return fig
    
    def plot_risk_prediction_gauge(self, risk_score):
        """Jauge de risque d'abandon"""
        # Définition des couleurs selon le niveau de risque
        if risk_score < 0.3:
            color = "green"
            risk_level = "Faible"
        elif risk_score < 0.7:
            color = "orange"
            risk_level = "Modéré"
        else:
            color = "red"
            risk_level = "Élevé"
        
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = risk_score * 100,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': f"Risque d'Abandon: {risk_level}"},
            delta = {'reference': 50},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': color},
                'steps': [
                    {'range': [0, 30], 'color': "lightgray"},
                    {'range': [30, 70], 'color': "yellow"},
                    {'range': [70, 100], 'color': "lightcoral"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        
        return fig
    
    def plot_association_rules_network(self, rules_df):
        """Visualisation réseau des règles d'association"""
        if rules_df is None or rules_df.empty:
            return None
        
        # Simplification pour l'affichage
        top_rules = rules_df.head(10)
        
        # Création d'un graphique en barres des règles les plus importantes
        fig = px.bar(
            top_rules,
            x='confidence',
            y=range(len(top_rules)),
            orientation='h',
            title="Top 10 des Règles d'Association",
            labels={'confidence': 'Confiance', 'y': 'Règles'}
        )
        
        # Personnalisation des labels Y
        fig.update_layout(
            yaxis=dict(
                tickmode='array',
                tickvals=list(range(len(top_rules))),
                ticktext=[f"Règle {i+1}" for i in range(len(top_rules))]
            )
        )
        
        return fig
    
    def create_comprehensive_dashboard(self, df, additional_data=None):
        """Création d'un dashboard complet"""
        dashboard_plots = {}
        
        # Graphiques principaux
        dashboard_plots['dropout_distribution'] = self.plot_dropout_distribution(df)
        dashboard_plots['feature_distributions'] = self.plot_feature_distributions(df)
        dashboard_plots['correlation_heatmap'] = self.plot_correlation_heatmap(df)
        
        # Graphiques par catégorie
        categorical_cols = ['gender', 'region', 'parent_education']
        for col in categorical_cols:
            if col in df.columns:
                dashboard_plots[f'dropout_by_{col}'] = self.plot_dropout_by_category(df, col)
        
        # Graphiques additionnels si données disponibles
        if additional_data:
            if 'clustered_data' in additional_data:
                dashboard_plots['clusters'] = self.plot_cluster_visualization(
                    additional_data['clustered_data']
                )
            
            if 'feature_importance' in additional_data:
                dashboard_plots['feature_importance'] = self.plot_feature_importance(
                    additional_data['feature_importance']
                )
            
            if 'association_rules' in additional_data:
                dashboard_plots['association_rules'] = self.plot_association_rules_network(
                    additional_data['association_rules']
                )
        
        return dashboard_plots

def create_visualizer():
    """Fonction pour créer une instance du visualiseur"""
    return DataVisualizer()