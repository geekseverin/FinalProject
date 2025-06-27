"""
Application Streamlit pour la prédiction d'abandon scolaire
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys
import warnings
warnings.filterwarnings('ignore')

# Ajout du chemin pour l'importation des modules
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent))

from config.config import *
from utils.visualization import DataVisualizer
from src.data_preprocessing import DataPreprocessor

# Configuration de la page
st.set_page_config(
    page_title=STREAMLIT_CONFIG['page_title'],
    page_icon=STREAMLIT_CONFIG['page_icon'],
    layout=STREAMLIT_CONFIG['layout']
)

class StudentDropoutApp:
    def __init__(self):
        self.visualizer = DataVisualizer()
        self.load_models()
        
    def load_models(self):
        """Chargement des modèles sauvegardés"""
        try:
            # Modèle de classification
            if (TRAINED_MODELS_DIR / "best_classifier.pkl").exists():
                self.classifier = joblib.load(TRAINED_MODELS_DIR / "best_classifier.pkl")
            else:
                self.classifier = None
                
            # Modèle de clustering
            if (CLUSTERING_DIR / "kmeans_model.pkl").exists():
                self.cluster_model = joblib.load(CLUSTERING_DIR / "kmeans_model.pkl")
            else:
                self.cluster_model = None
                
            # Objets de preprocessing
            if (MODELS_DIR / "scaler.pkl").exists():
                self.scaler = joblib.load(MODELS_DIR / "scaler.pkl")
            else:
                self.scaler = None
                
            if (MODELS_DIR / "label_encoders.pkl").exists():
                self.label_encoders = joblib.load(MODELS_DIR / "label_encoders.pkl")
            else:
                self.label_encoders = {}
                
        except Exception as e:
            st.error(f"Erreur lors du chargement des modèles: {e}")
            self.classifier = None
            self.cluster_model = None
            self.scaler = None
            self.label_encoders = {}
    
    def load_data(self):
        """Chargement des données"""
        try:
            # Données nettoyées
            if (PROCESSED_DATA_DIR / "cleaned_data.csv").exists():
                self.data = pd.read_csv(PROCESSED_DATA_DIR / "cleaned_data.csv")
            else:
                self.data = None
                
            # Données avec clusters
            if (OUTPUTS_DIR / "cluster_analysis.csv").exists():
                self.cluster_analysis = pd.read_csv(OUTPUTS_DIR / "cluster_analysis.csv")
            else:
                self.cluster_analysis = None
                
            # Importance des variables
            if (OUTPUTS_DIR / "feature_importance.csv").exists():
                self.feature_importance = pd.read_csv(OUTPUTS_DIR / "feature_importance.csv")
            else:
                self.feature_importance = None
                
        except Exception as e:
            st.error(f"Erreur lors du chargement des données: {e}")
            self.data = None
    
    def sidebar_navigation(self):
        """Navigation latérale"""
        st.sidebar.title("🎓 Navigation")
        
        pages = {
            "🏠 Accueil": "home",
            "📊 Tableau de Bord": "dashboard", 
            "🔮 Prédiction": "prediction",
            "👥 Analyse de Clustering": "clustering",
            "📋 Règles d'Association": "association",
            "⚙️ Configuration": "settings"
        }
        
        selected_page = st.sidebar.selectbox("Choisir une page", list(pages.keys()))
        return pages[selected_page]
    
    def home_page(self):
        """Page d'accueil"""
        st.title("🎓 Système de Prédiction d'Abandon Scolaire")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("📚 Étudiants Analysés", 
                     len(self.data) if self.data is not None else "N/A")
        
        with col2:
            if self.data is not None:
                dropout_rate = self.data['dropout'].mean() * 100
                st.metric("⚠️ Taux d'Abandon", f"{dropout_rate:.1f}%")
            else:
                st.metric("⚠️ Taux d'Abandon", "N/A")
        
        with col3:
            st.metric("🤖 Modèles Disponibles", 
                     "Actif" if self.classifier is not None else "Inactif")
        
        st.markdown("""
        ## 🎯 Objectif du Projet
        
        Ce système utilise l'intelligence artificielle pour prédire le risque d'abandon scolaire 
        des étudiants en se basant sur diverses caractéristiques académiques et comportementales.
        
        ### 📋 Fonctionnalités
        
        - **Prédiction Individuelle**: Prédire le risque d'abandon pour un étudiant spécifique
        - **Analyse de Clusters**: Identifier les profils d'étudiants similaires
        - **Règles d'Association**: Découvrir les patterns menant à l'abandon
        - **Tableau de Bord**: Visualisations interactives des données
        
        ### 🔧 Technologies
        
        - **Machine Learning**: Scikit-learn, XGBoost
        - **Visualisation**: Plotly, Streamlit
        - **Clustering**: K-Means, DBSCAN
        - **Association**: Algorithme Apriori
        """)
        
        if self.data is not None:
            st.subheader("📊 Aperçu des Données")
            st.dataframe(self.data.head())
    
    def dashboard_page(self):
        """Page du tableau de bord"""
        st.title("📊 Tableau de Bord Analytique")
        
        if self.data is None:
            st.error("Aucune donnée disponible. Veuillez d'abord exécuter le pipeline ETL.")
            return
        
        # Distribution de l'abandon
        st.subheader("Distribution de l'Abandon Scolaire")
        fig_dropout = self.visualizer.plot_dropout_distribution(self.data)
        st.plotly_chart(fig_dropout, use_container_width=True)
        
        # Statistiques par catégorie
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Abandon par Genre")
            if 'gender' in self.data.columns:
                fig_gender = self.visualizer.plot_dropout_by_category(self.data, 'gender')
                st.plotly_chart(fig_gender, use_container_width=True)
        
        with col2:
            st.subheader("Abandon par Région")
            if 'region' in self.data.columns:
                fig_region = self.visualizer.plot_dropout_by_category(self.data, 'region')
                st.plotly_chart(fig_region, use_container_width=True)
        
        # Corrélations
        st.subheader("Matrice de Corrélation")
        fig_corr = self.visualizer.plot_correlation_heatmap(self.data)
        st.plotly_chart(fig_corr, use_container_width=True)
        
        # Importance des variables
        if self.feature_importance is not None:
            st.subheader("Importance des Variables")
            fig_importance = self.visualizer.plot_feature_importance(self.feature_importance)
            st.plotly_chart(fig_importance, use_container_width=True)
    
    def prediction_page(self):
        """Page de prédiction"""
        st.title("🔮 Prédiction d'Abandon Scolaire")
        
        if self.classifier is None:
            st.error("Modèle de classification non disponible. Veuillez d'abord entraîner les modèles.")
            return
        
        st.markdown("### Saisir les informations de l'étudiant")
        
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.number_input("Âge", min_value=18, max_value=50, value=22)
            gender = st.selectbox("Genre", ["Male", "Female"])
            region = st.selectbox("Région", ["Lome", "Notse", "Tsevie", "Dapaong", "Sokode"])
            parent_education = st.selectbox("Éducation des Parents", 
                                          ["None", "Primary", "Secondary", "Higher"])
            average_grade = st.number_input("Note Moyenne", min_value=0.0, max_value=20.0, value=12.0)
        
        with col2:
            absenteeism_rate = st.number_input("Taux d'Absentéisme (%)", 
                                             min_value=0.0, max_value=100.0, value=10.0)
            assignments_submitted = st.number_input("Devoirs Rendus (%)", 
                                                  min_value=0.0, max_value=100.0, value=80.0)
            moodle_hours = st.number_input("Heures Moodle", min_value=0.0, max_value=50.0, value=10.0)
            forum_posts = st.number_input("Posts Forum", min_value=0, max_value=20, value=3)
            satisfaction_score = st.number_input("Score de Satisfaction", 
                                               min_value=1.0, max_value=10.0, value=7.0)
        
        if st.button("🚀 Prédire le Risque d'Abandon", type="primary"):
            # Préparation des données
            input_data = self.prepare_prediction_data(
                age, gender, region, parent_education, average_grade,
                absenteeism_rate, assignments_submitted, moodle_hours,
                forum_posts, satisfaction_score
            )
            
            if input_data is not None:
                # Prédiction
                risk_proba = self.classifier.predict_proba(input_data)[0][1]
                
                # Affichage du résultat
                st.subheader("📈 Résultat de la Prédiction")
                
                # Jauge de risque
                fig_gauge = self.visualizer.plot_risk_prediction_gauge(risk_proba)
                st.plotly_chart(fig_gauge, use_container_width=True)
                
                # Recommandations
                st.subheader("💡 Recommandations")
                recommendations = self.generate_recommendations(risk_proba, input_data)
                for rec in recommendations:
                    st.write(f"• {rec}")
    
    def prepare_prediction_data(self, age, gender, region, parent_education, 
                              average_grade, absenteeism_rate, assignments_submitted,
                              moodle_hours, forum_posts, satisfaction_score):
        """Préparation des données pour la prédiction"""
        try:
            # Création du DataFrame
            data = pd.DataFrame({
                'age': [age],
                'gender': [gender],
                'region': [region], 
                'parent_education': [parent_education],
                'average_grade': [average_grade],
                'absenteeism_rate': [absenteeism_rate],
                'assignments_submitted': [assignments_submitted],
                'moodle_hours': [moodle_hours],
                'forum_posts': [forum_posts],
                'satisfaction_score': [satisfaction_score]
            })
            
            # Encodage des variables catégorielles
            for col in ['gender', 'region', 'parent_education']:
                if col in self.label_encoders:
                    try:
                        data[col] = self.label_encoders[col].transform(data[col])
                    except ValueError:
                        # Valeur non vue pendant l'entraînement
                        data[col] = 0
            
            # Feature engineering (mêmes transformations qu'à l'entraînement)
            data['participation_ratio'] = data['forum_posts'] / (data['moodle_hours'] + 1)
            data['engagement_score'] = (
                data['assignments_submitted'] * 0.4 + 
                data['moodle_hours'] * 0.3 + 
                data['forum_posts'] * 0.3
            )
            
            # Catégorie de satisfaction
            satisfaction_cat = 'Low' if satisfaction_score <= 3 else ('Medium' if satisfaction_score <= 6 else 'High')
            if 'satisfaction_category' in self.label_encoders:
                try:
                    data['satisfaction_category'] = self.label_encoders['satisfaction_category'].transform([satisfaction_cat])
                except ValueError:
                    data['satisfaction_category'] = 0
            else:
                data['satisfaction_category'] = 0
            
            # Standardisation
            if self.scaler is not None:
                # Sélection des colonnes à standardiser
                cols_to_scale = [col for col in data.columns if data[col].dtype in ['int64', 'float64']]
                data[cols_to_scale] = self.scaler.transform(data[cols_to_scale])
            
            return data
            
        except Exception as e:
            st.error(f"Erreur lors de la préparation des données: {e}")
            return None
    
    def generate_recommendations(self, risk_proba, input_data):
        """Génération de recommandations personnalisées"""
        recommendations = []
        
        if risk_proba > 0.7:
            recommendations.append("🚨 RISQUE ÉLEVÉ: Intervention immédiate recommandée")
            recommendations.append("📞 Contacter l'étudiant pour un entretien personnalisé")
            recommendations.append("🎯 Mettre en place un plan d'accompagnement individualisé")
        elif risk_proba > 0.4:
            recommendations.append("⚠️ RISQUE MODÉRÉ: Surveillance renforcée nécessaire")
            recommendations.append("📧 Envoyer des rappels réguliers et encouragements")
        else:
            recommendations.append("✅ RISQUE FAIBLE: Continuer le suivi habituel")
        
        # Recommandations spécifiques basées sur les données
        if input_data['absenteeism_rate'].iloc[0] > 0.2:  # > 20% après standardisation
            recommendations.append("📅 Améliorer l'assiduité aux cours")
        
        if input_data['satisfaction_score'].iloc[0] < -0.5:  # Score faible après standardisation
            recommendations.append("😊 Améliorer l'expérience étudiante")
            
        if input_data['forum_posts'].iloc[0] < -0.5:  # Peu de participation
            recommendations.append("💬 Encourager la participation aux forums")
        
        return recommendations
    
    def clustering_page(self):
        """Page d'analyse de clustering"""
        st.title("👥 Analyse de Clustering")
        
        if self.cluster_analysis is not None:
            st.subheader("📊 Profils des Clusters")
            st.dataframe(self.cluster_analysis)
            
            # Visualisation des clusters
            if self.data is not None:
                st.subheader("🎯 Visualisation des Clusters")
                # Simulation de données de clusters pour la visualisation
                if 'cluster' not in self.data.columns:
                    # Ajout temporaire de clusters aléatoires pour la démo
                    self.data['cluster'] = np.random.randint(0, 3, len(self.data))
                
                fig_clusters = self.visualizer.plot_cluster_visualization(self.data)
                if fig_clusters:
                    st.plotly_chart(fig_clusters, use_container_width=True)
        else:
            st.info("Aucune analyse de clustering disponible. Veuillez d'abord exécuter l'analyse.")
    
    def association_page(self):
        """Page des règles d'association"""
        st.title("📋 Règles d'Association")
        
        # Chargement des règles d'association
        if (ASSOCIATION_RULES_DIR / "recommendations.pkl").exists():
            recommendations = joblib.load(ASSOCIATION_RULES_DIR / "recommendations.pkl")
            
            st.subheader("🎯 Patterns à Risque Identifiés")
            if 'high_risk_patterns' in recommendations:
                for i, pattern in enumerate(recommendations['high_risk_patterns'][:5]):
                    with st.expander(f"Pattern {i+1} - Risque: {pattern.get('dropout_risk', 0):.2f}"):
                        st.write("**Conditions:**")
                        for condition in pattern.get('conditions', []):
                            st.write(f"• {condition}")
                        st.write(f"**Force:** {pattern.get('strength', 0):.2f}")
            
            st.subheader("💡 Suggestions d'Intervention")
            if 'intervention_suggestions' in recommendations:
                for suggestion in recommendations['intervention_suggestions'][:10]:
                    st.write(f"• {suggestion}")
                    
        else:
            st.info("Aucune règle d'association disponible. Veuillez d'abord exécuter l'analyse.")
    
    def settings_page(self):
        """Page de configuration"""
        st.title("⚙️ Configuration du Système")
        
        st.subheader("📁 État des Fichiers")
        
        files_status = {
            "Données Brutes": (RAW_DATA_DIR / "student_data.csv").exists(),
            "Données Nettoyées": (PROCESSED_DATA_DIR / "cleaned_data.csv").exists(),
            "Modèle Classificateur": (TRAINED_MODELS_DIR / "best_classifier.pkl").exists(),
            "Modèle Clustering": (CLUSTERING_DIR / "kmeans_model.pkl").exists(),
            "Scaler": (MODELS_DIR / "scaler.pkl").exists()
        }
        
        for file_name, exists in files_status.items():
            status = "✅" if exists else "❌"
            st.write(f"{status} {file_name}")
        
        st.subheader("🔧 Actions Disponibles")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🔄 Recharger les Modèles"):
                self.load_models()
                st.success("Modèles rechargés!")
        
        with col2:
            if st.button("📊 Recharger les Données"):
                self.load_data()
                st.success("Données rechargées!")
        
        with col3:
            if st.button("🧹 Nettoyer le Cache"):
                st.cache_data.clear()
                st.success("Cache nettoyé!")
    
    def run(self):
        """Lancement de l'application"""
        # Chargement des données
        self.load_data()
        
        # Navigation
        page = self.sidebar_navigation()
        
        # Affichage de la page sélectionnée
        if page == "home":
            self.home_page()
        elif page == "dashboard":
            self.dashboard_page()
        elif page == "prediction":
            self.prediction_page()
        elif page == "clustering":
            self.clustering_page()
        elif page == "association":
            self.association_page()
        elif page == "settings":
            self.settings_page()

def main():
    """Fonction principale"""
    app = StudentDropoutApp()
    app.run()

if __name__ == "__main__":
    main()