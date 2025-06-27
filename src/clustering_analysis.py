"""
Module d'analyse de clustering pour identifier les profils d'étudiants
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import joblib
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))
from config.config import *

class ClusteringAnalyzer:
    def __init__(self):
        self.kmeans_models = {}
        self.dbscan_model = None
        self.best_kmeans = None
        self.best_n_clusters = 0
        self.best_silhouette = -1
        self.pca_model = None
        
    def load_processed_data(self):
        """Charge les données préprocessées"""
        try:
            data_path = PROCESSED_DATA_DIR / "scaled_data.csv"
            return pd.read_csv(data_path)
        except Exception as e:
            print(f"Erreur lors du chargement des données: {e}")
            return None
    
    def prepare_clustering_data(self, df):
        """Préparation des données pour le clustering"""
        # Exclusion des colonnes ID et target
        X = df.drop([ID_COLUMN, TARGET_COLUMN], axis=1)
        
        # Application de PCA pour la visualisation
        self.pca_model = PCA(n_components=2, random_state=MODEL_CONFIG['random_state'])
        X_pca = self.pca_model.fit_transform(X)
        
        return X, X_pca
    
    def find_optimal_clusters_kmeans(self, X):
        """Trouve le nombre optimal de clusters pour K-Means"""
        print("🔄 Recherche du nombre optimal de clusters...")
        
        silhouette_scores = []
        
        for n_clusters in MODEL_CONFIG['n_clusters_range']:
            kmeans = KMeans(
                n_clusters=n_clusters, 
                random_state=MODEL_CONFIG['random_state'],
                n_init=10
            )
            cluster_labels = kmeans.fit_predict(X)
            
            # Calcul du score de silhouette
            silhouette_avg = silhouette_score(X, cluster_labels)
            silhouette_scores.append(silhouette_avg)
            
            # Sauvegarde du modèle
            self.kmeans_models[n_clusters] = kmeans
            
            # Mise à jour du meilleur modèle
            if silhouette_avg > self.best_silhouette:
                self.best_silhouette = silhouette_avg
                self.best_kmeans = kmeans
                self.best_n_clusters = n_clusters
            
            print(f"K={n_clusters}, Silhouette Score: {silhouette_avg:.4f}")
        
        return silhouette_scores
    
    def apply_dbscan(self, X):
        """Application de DBSCAN"""
        print("🔄 Application de DBSCAN...")
        
        # Test de différents paramètres eps
        eps_values = [0.5, 1.0, 1.5, 2.0]
        best_eps = 0.5
        best_dbscan_score = -1
        
        for eps in eps_values:
            dbscan = DBSCAN(eps=eps, min_samples=5)
            cluster_labels = dbscan.fit_predict(X)
            
            # Vérification qu'il y a au moins 2 clusters
            n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
            
            if n_clusters > 1:
                # Exclusion des points de bruit pour le calcul du silhouette
                mask = cluster_labels != -1
                if np.sum(mask) > 0:
                    silhouette_avg = silhouette_score(X[mask], cluster_labels[mask])
                    
                    if silhouette_avg > best_dbscan_score:
                        best_dbscan_score = silhouette_avg
                        best_eps = eps
                        self.dbscan_model = dbscan
                    
                    print(f"EPS={eps}, Clusters: {n_clusters}, Silhouette: {silhouette_avg:.4f}")
        
        # Entraînement final avec le meilleur eps
        if self.dbscan_model is None:
            self.dbscan_model = DBSCAN(eps=best_eps, min_samples=5)
            self.dbscan_model.fit(X)
        
        return self.dbscan_model
    
    def analyze_clusters(self, df, X):
        """Analyse des profils de clusters"""
        print("🔄 Analyse des profils de clusters...")
        
        # Prédiction des clusters avec le meilleur modèle K-Means
        cluster_labels = self.best_kmeans.predict(X)
        
        # Ajout des labels au DataFrame original
        df_with_clusters = df.copy()
        df_with_clusters['cluster'] = cluster_labels
        
        # Analyse par cluster
        cluster_analysis = {}
        
        for i in range(self.best_n_clusters):
            cluster_data = df_with_clusters[df_with_clusters['cluster'] == i]
            
            analysis = {
                'size': len(cluster_data),
                'dropout_rate': cluster_data[TARGET_COLUMN].mean(),
                'avg_grade': cluster_data['average_grade'].mean(),
                'avg_absenteeism': cluster_data['absenteeism_rate'].mean(),
                'avg_satisfaction': cluster_data['satisfaction_score'].mean(),
                'avg_forum_posts': cluster_data['forum_posts'].mean(),
                'avg_moodle_hours': cluster_data['moodle_hours'].mean()
            }
            
            cluster_analysis[f'cluster_{i}'] = analysis
        
        # Sauvegarde de l'analyse
        cluster_df = pd.DataFrame(cluster_analysis).T
        cluster_df.to_csv(OUTPUTS_DIR / "cluster_analysis.csv")
        
        return cluster_analysis, df_with_clusters
    
    def save_clustering_models(self):
        """Sauvegarde des modèles de clustering"""
        # Sauvegarde du meilleur K-Means
        joblib.dump(self.best_kmeans, CLUSTERING_DIR / "kmeans_model.pkl")
        
        # Sauvegarde de DBSCAN
        if self.dbscan_model:
            joblib.dump(self.dbscan_model, CLUSTERING_DIR / "dbscan_model.pkl")
        
        # Sauvegarde du modèle PCA
        joblib.dump(self.pca_model, CLUSTERING_DIR / "pca_model.pkl")
        
        # Sauvegarde des informations sur le clustering
        clustering_info = {
            'best_n_clusters': self.best_n_clusters,
            'best_silhouette_score': self.best_silhouette,
            'pca_explained_variance': self.pca_model.explained_variance_ratio_.tolist()
        }
        
        joblib.dump(clustering_info, CLUSTERING_DIR / "clustering_info.pkl")
    
    def run_clustering_analysis(self):
        """Pipeline complet d'analyse de clustering"""
        print("🔄 Début de l'analyse de clustering...")
        
        # Chargement des données
        df = self.load_processed_data()
        if df is None:
            return None
        
        # Préparation des données
        X, X_pca = self.prepare_clustering_data(df)
        print(f"✅ Données préparées: {X.shape}")
        
        # Recherche optimal K-Means
        silhouette_scores = self.find_optimal_clusters_kmeans(X)
        
        # Application de DBSCAN
        self.apply_dbscan(X)
        
        # Analyse des clusters
        cluster_analysis, df_with_clusters = self.analyze_clusters(df, X)
        
        # Sauvegarde des modèles
        self.save_clustering_models()
        
        print(f"✅ Clustering terminé! Meilleur K={self.best_n_clusters}, Silhouette={self.best_silhouette:.4f}")
        
        return {
            'best_n_clusters': self.best_n_clusters,
            'best_silhouette_score': self.best_silhouette,
            'cluster_analysis': cluster_analysis,
            'silhouette_scores': silhouette_scores,
            'clustered_data': df_with_clusters
        }

def run_clustering():
    """Fonction principale pour l'analyse de clustering"""
    analyzer = ClusteringAnalyzer()
    return analyzer.run_clustering_analysis()

if __name__ == "__main__":
    run_clustering()