"""
Module d'ingénierie des variables pour la prédiction de l'abandon scolaire
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.ensemble import RandomForestClassifier
import joblib
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))
from config.config import *

class FeatureEngineer:
    def __init__(self):
        self.poly_features = None
        self.pca_model = None
        self.feature_selector = None
        self.rf_selector = None
        
    def load_cleaned_data(self):
        """Charge les données nettoyées"""
        try:
            data_path = PROCESSED_DATA_DIR / "cleaned_data.csv"
            return pd.read_csv(data_path)
        except Exception as e:
            print(f"❌ Erreur lors du chargement des données: {e}")
            return None
    
    def create_academic_features(self, df):
        """Création de variables académiques avancées"""
        print("🔄 Création des variables académiques...")
        
        df = df.copy()
        
        # Ratio de performance académique
        df['performance_ratio'] = df['average_grade'] / df['age']
        
        # Score d'assiduité inversé
        df['attendance_score'] = 100 - df['absenteeism_rate']
        
        # Taux de completion des devoirs
        df['assignment_completion_rate'] = df['assignments_submitted'] / df['age']  # Normalisation par âge
        
        # Indicateur de performance critique
        df['critical_performance'] = ((df['average_grade'] < 10) & 
                                     (df['absenteeism_rate'] > 20)).astype(int)
        
        # Catégorie de notes
        df['grade_category'] = pd.cut(df['average_grade'], 
                                    bins=[0, 8, 12, 16, 20], 
                                    labels=['Faible', 'Moyen', 'Bon', 'Excellent'])
        
        # Binning de l'absentéisme
        df['absenteeism_category'] = pd.cut(df['absenteeism_rate'],
                                          bins=[0, 5, 15, 25, 100],
                                          labels=['Très_faible', 'Faible', 'Modéré', 'Élevé'])
        
        return df
    
    def create_engagement_features(self, df):
        """Création de variables d'engagement"""
        print("🔄 Création des variables d'engagement...")
        
        df = df.copy()
        
        # Score d'engagement total
        df['total_engagement'] = (df['moodle_hours'] + 
                                 df['forum_posts'] + 
                                 df['assignments_submitted'])
        
        # Ratio forum/moodle (interaction vs consommation)
        df['interaction_ratio'] = df['forum_posts'] / (df['moodle_hours'] + 1)
        
        # Intensité d'usage de Moodle
        df['moodle_intensity'] = pd.cut(df['moodle_hours'],
                                       bins=[0, 5, 15, 30, 100],
                                       labels=['Faible', 'Modéré', 'Élevé', 'Très_élevé'])
        
        # Participation aux forums
        df['forum_participation'] = pd.cut(df['forum_posts'],
                                         bins=[0, 1, 5, 15, 100],
                                         labels=['Aucune', 'Faible', 'Modérée', 'Active'])
        
        # Score d'engagement digital
        df['digital_engagement_score'] = (
            (df['moodle_hours'] / df['moodle_hours'].max()) * 0.4 +
            (df['forum_posts'] / df['forum_posts'].max()) * 0.3 +
            (df['assignments_submitted'] / df['assignments_submitted'].max()) * 0.3
        )
        
        # Indicateur d'engagement faible
        df['low_engagement'] = (
            (df['moodle_hours'] < df['moodle_hours'].quantile(0.25)) &
            (df['forum_posts'] < df['forum_posts'].quantile(0.25))
        ).astype(int)
        
        return df
    
    def create_satisfaction_features(self, df):
        """Création de variables de satisfaction"""
        print("🔄 Création des variables de satisfaction...")
        
        df = df.copy()
        
        # Catégorie de satisfaction
        df['satisfaction_category'] = pd.cut(df['satisfaction_score'],
                                           bins=[0, 4, 7, 10],
                                           labels=['Insatisfait', 'Neutre', 'Satisfait'])
        
        # Indicateur de satisfaction critique
        df['critical_satisfaction'] = (df['satisfaction_score'] <= 3).astype(int)
        
        # Score de satisfaction normalisé
        df['satisfaction_normalized'] = df['satisfaction_score'] / 10
        
        # Combinaison satisfaction-performance
        df['satisfaction_performance'] = df['satisfaction_score'] * df['average_grade']
        
        return df
    
    def create_demographic_features(self, df):
        """Création de variables démographiques avancées"""
        print("🔄 Création des variables démographiques...")
        
        df = df.copy()
        
        # Catégorie d'âge
        df['age_category'] = pd.cut(df['age'],
                                  bins=[0, 20, 25, 30, 100],
                                  labels=['Très_jeune', 'Jeune', 'Adulte', 'Mature'])
        
        # Indicateur de première génération universitaire
        df['first_generation'] = (df['parent_education'] == 'None').astype(int)
        
        # Interaction âge-région (peut indiquer des patterns géographiques)
        df['age_region_interaction'] = df['age'].astype(str) + '_' + df['region'].astype(str)
        
        return df
    
    def create_interaction_features(self, df):
        """Création de variables d'interaction"""
        print("🔄 Création des variables d'interaction...")
        
        df = df.copy()
        
        # Interaction performance-satisfaction
        df['performance_satisfaction'] = df['average_grade'] * df['satisfaction_score']
        
        # Interaction engagement-performance
        df['engagement_performance'] = df['total_engagement'] * df['average_grade']
        
        # Interaction absentéisme-satisfaction
        df['absence_satisfaction'] = df['absenteeism_rate'] * (10 - df['satisfaction_score'])
        
        # Ratio performance/âge
        df['performance_age_ratio'] = df['average_grade'] / df['age']
        
        return df
    
    def create_risk_indicators(self, df):
        """Création d'indicateurs de risque"""
        print("🔄 Création des indicateurs de risque...")
        
        df = df.copy()
        
        # Score de risque composite
        df['risk_score'] = (
            (df['absenteeism_rate'] > 20).astype(int) * 0.3 +
            (df['average_grade'] < 10).astype(int) * 0.3 +
            (df['satisfaction_score'] < 5).astype(int) * 0.2 +
            (df['total_engagement'] < df['total_engagement'].quantile(0.25)).astype(int) * 0.2
        )
        
        # Indicateurs de risque binaires
        df['high_absence_risk'] = (df['absenteeism_rate'] > 25).astype(int)
        df['low_performance_risk'] = (df['average_grade'] < 8).astype(int)
        df['low_satisfaction_risk'] = (df['satisfaction_score'] < 4).astype(int)
        df['low_engagement_risk'] = (df['total_engagement'] < 10).astype(int)
        
        # Nombre de facteurs de risque
        df['risk_factors_count'] = (
            df['high_absence_risk'] + 
            df['low_performance_risk'] + 
            df['low_satisfaction_risk'] + 
            df['low_engagement_risk']
        )
        
        return df
    
    def encode_categorical_features(self, df):
        """Encodage des nouvelles variables catégorielles"""
        print("🔄 Encodage des variables catégorielles...")
        
        df = df.copy()
        
        # Encodage one-hot pour les nouvelles variables catégorielles
        categorical_features = [
            'grade_category', 'absenteeism_category', 'moodle_intensity',
            'forum_participation', 'satisfaction_category', 'age_category'
        ]
        
        for feature in categorical_features:
            if feature in df.columns:
                # One-hot encoding
                dummies = pd.get_dummies(df[feature], prefix=feature, drop_first=True)
                df = pd.concat([df, dummies], axis=1)
                df.drop(feature, axis=1, inplace=True)
        
        return df
    
    def create_polynomial_features(self, df, degree=2):
        """Création de variables polynomiales"""
        print("🔄 Création des variables polynomiales...")
        
        # Sélection des variables numériques pour les polynômes
        numeric_features = ['average_grade', 'satisfaction_score', 'total_engagement']
        
        if all(col in df.columns for col in numeric_features):
            self.poly_features = PolynomialFeatures(degree=degree, 
                                                   include_bias=False, 
                                                   interaction_only=True)
            
            poly_data = self.poly_features.fit_transform(df[numeric_features])
            poly_feature_names = self.poly_features.get_feature_names_out(numeric_features)
            
            # Création du DataFrame avec les nouvelles variables
            poly_df = pd.DataFrame(poly_data, columns=poly_feature_names, index=df.index)
            
            # Suppression des colonnes originales pour éviter la duplication
            poly_df = poly_df.drop(columns=numeric_features)
            
            # Ajout des nouvelles variables
            df = pd.concat([df, poly_df], axis=1)
        
        return df
    
    def apply_feature_selection(self, df):
        """Sélection des variables les plus importantes"""
        print("🔄 Sélection des variables importantes...")
        
        # Séparation des features et du target
        X = df.drop([ID_COLUMN, TARGET_COLUMN], axis=1)
        y = df[TARGET_COLUMN]
        
        # Sélection univariée
        self.feature_selector = SelectKBest(score_func=f_classif, k=min(20, X.shape[1]))
        X_selected = self.feature_selector.fit_transform(X, y)
        
        # Récupération des noms des features sélectionnées
        selected_features = X.columns[self.feature_selector.get_support()]
        
        # Sélection par Random Forest
        rf = RandomForestClassifier(n_estimators=100, random_state=MODEL_CONFIG['random_state'])
        self.rf_selector = RFE(rf, n_features_to_select=min(15, len(selected_features)))
        self.rf_selector.fit(X[selected_features], y)
        
        # Features finales
        final_features = selected_features[self.rf_selector.get_support()]
        
        # Création du DataFrame final
        final_df = df[[ID_COLUMN, TARGET_COLUMN]].copy()
        final_df = pd.concat([final_df, df[final_features]], axis=1)
        
        print(f"✅ Sélection terminée: {len(final_features)} variables retenues")
        return final_df, final_features
    
    def save_feature_engineering_objects(self):
        """Sauvegarde des objets de feature engineering"""
        if self.poly_features:
            joblib.dump(self.poly_features, MODELS_DIR / "poly_features.pkl")
        
        if self.feature_selector:
            joblib.dump(self.feature_selector, MODELS_DIR / "feature_selector.pkl")
        
        if self.rf_selector:
            joblib.dump(self.rf_selector, MODELS_DIR / "rf_selector.pkl")
    
    def run_feature_engineering(self):
        """Pipeline complet de feature engineering"""
        print("🔄 Début du feature engineering...")
        
        # Chargement des données
        df = self.load_cleaned_data()
        if df is None:
            return None
        
        print(f"✅ Données chargées: {df.shape}")
        
        # Création des différents types de variables
        df = self.create_academic_features(df)
        df = self.create_engagement_features(df)
        df = self.create_satisfaction_features(df)
        df = self.create_demographic_features(df)
        df = self.create_interaction_features(df)
        df = self.create_risk_indicators(df)
        
        print(f"✅ Variables créées: {df.shape}")
        
        # Encodage des variables catégorielles
        df = self.encode_categorical_features(df)
        
        print(f"✅ Encodage terminé: {df.shape}")
        
        # Création de variables polynomiales
        df = self.create_polynomial_features(df)
        
        print(f"✅ Variables polynomiales créées: {df.shape}")
        
        # Sélection des variables importantes
        final_df, selected_features = self.apply_feature_selection(df)
        
        # Sauvegarde des résultats
        final_df.to_csv(PROCESSED_DATA_DIR / "engineered_features.csv", index=False)
        
        # Sauvegarde des objets
        self.save_feature_engineering_objects()
        
        # Sauvegarde de la liste des variables sélectionnées
        pd.DataFrame({'selected_features': selected_features}).to_csv(
            OUTPUTS_DIR / "selected_features.csv", index=False
        )
        
        print("✅ Feature engineering terminé avec succès!")
        
        return {
            'engineered_data': final_df,
            'selected_features': selected_features,
            'n_features_created': df.shape[1] - len(ALL_COLUMNS),
            'n_features_selected': len(selected_features)
        }

def run_feature_engineering():
    """Fonction principale pour le feature engineering"""
    engineer = FeatureEngineer()
    return engineer.run_feature_engineering()

if __name__ == "__main__":
    run_feature_engineering()