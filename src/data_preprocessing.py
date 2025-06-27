"""
Module de préprocessing des données
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import joblib
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))
from config.config import *

class DataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.imputers = {}
        
    def load_raw_data(self):
        """Charge les données brutes"""
        try:
            data_path = RAW_DATA_DIR / "student_data.csv"
            return pd.read_csv(data_path)
        except Exception as e:
            print(f"Erreur lors du chargement des données: {e}")
            return None
    
    def clean_data(self, df):
        """Nettoyage initial des données"""
        df = df.copy()
        
        # Suppression des doublons
        df = df.drop_duplicates()
        
        # Conversion des types
        df['dropout'] = df['dropout'].map({'Yes': 1, 'No': 0})
        
        # Gestion des valeurs manquantes pour les variables numériques
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col not in [ID_COLUMN, TARGET_COLUMN]:
                self.imputers[col] = SimpleImputer(strategy='median')
                df[col] = self.imputers[col].fit_transform(df[[col]]).flatten()
        
        # Gestion des valeurs manquantes pour les variables catégorielles
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if col != TARGET_COLUMN:
                self.imputers[col] = SimpleImputer(strategy='most_frequent')
                df[col] = self.imputers[col].fit_transform(df[[col]]).flatten()
        
        return df
    
    def encode_categorical_features(self, df):
        """Encodage des variables catégorielles"""
        df = df.copy()
        categorical_cols = ['gender', 'region', 'parent_education']
        
        for col in categorical_cols:
            if col in df.columns:
                self.label_encoders[col] = LabelEncoder()
                df[col] = self.label_encoders[col].fit_transform(df[col])
        
        return df
    
    def create_features(self, df):
        """Création de nouvelles variables"""
        df = df.copy()
        
        # Ratio participation/temps
        df['participation_ratio'] = df['forum_posts'] / (df['moodle_hours'] + 1)
        
        # Score d'engagement
        df['engagement_score'] = (
            df['assignments_submitted'] * 0.4 + 
            df['moodle_hours'] * 0.3 + 
            df['forum_posts'] * 0.3
        )
        
        # Catégorie de satisfaction
        df['satisfaction_category'] = pd.cut(df['satisfaction_score'], 
                                           bins=[0, 3, 6, 10], 
                                           labels=['Low', 'Medium', 'High'])
        
        # Encodage de la nouvelle variable catégorielle
        if 'satisfaction_category' in df.columns:
            self.label_encoders['satisfaction_category'] = LabelEncoder()
            df['satisfaction_category'] = self.label_encoders['satisfaction_category'].fit_transform(df['satisfaction_category'])
        
        return df
    
    def scale_features(self, df):
        """Standardisation des variables numériques"""
        df = df.copy()
        
        # Colonnes à standardiser (exclure ID et target)
        cols_to_scale = [col for col in df.columns 
                        if col not in [ID_COLUMN, TARGET_COLUMN] 
                        and df[col].dtype in ['int64', 'float64']]
        
        df[cols_to_scale] = self.scaler.fit_transform(df[cols_to_scale])
        
        return df
    
    def save_preprocessors(self):
        """Sauvegarde des objets de preprocessing"""
        joblib.dump(self.scaler, MODELS_DIR / "scaler.pkl")
        joblib.dump(self.label_encoders, MODELS_DIR / "label_encoders.pkl")
        joblib.dump(self.imputers, MODELS_DIR / "imputers.pkl")
    
    def process_pipeline(self):
        """Pipeline complet de preprocessing"""
        print("🔄 Début du preprocessing des données...")
        
        # Chargement des données
        df = self.load_raw_data()
        if df is None:
            return None
        
        print(f"✅ Données chargées: {df.shape}")
        
        # Nettoyage
        df = self.clean_data(df)
        print("✅ Nettoyage terminé")
        
        # Sauvegarde des données nettoyées
        df.to_csv(PROCESSED_DATA_DIR / "cleaned_data.csv", index=False)
        
        # Encodage
        df = self.encode_categorical_features(df)
        print("✅ Encodage terminé")
        
        # Création de nouvelles variables
        df = self.create_features(df)
        print("✅ Feature engineering terminé")
        
        # Sauvegarde avec nouvelles variables
        df.to_csv(PROCESSED_DATA_DIR / "features_engineered.csv", index=False)
        
        # Standardisation
        df_scaled = self.scale_features(df)
        print("✅ Standardisation terminée")
        
        # Sauvegarde finale
        df_scaled.to_csv(PROCESSED_DATA_DIR / "scaled_data.csv", index=False)
        
        # Sauvegarde des objets de preprocessing
        self.save_preprocessors()
        print("✅ Preprocessing terminé avec succès!")
        
        return df_scaled

def run_preprocessing():
    """Fonction principale pour exécuter le preprocessing"""
    preprocessor = DataPreprocessor()
    return preprocessor.process_pipeline()

if __name__ == "__main__":
    run_preprocessing()