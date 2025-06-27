"""
Module d'entraînement des modèles de classification
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import joblib
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))
from config.config import *

class ModelTrainer:
    def __init__(self):
        self.models = {}
        self.best_model = None
        self.best_score = 0
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
    def load_processed_data(self):
        """Charge les données préprocessées"""
        try:
            data_path = PROCESSED_DATA_DIR / "scaled_data.csv"
            return pd.read_csv(data_path)
        except Exception as e:
            print(f"Erreur lors du chargement des données: {e}")
            return None
    
    def prepare_data(self, df):
        """Préparation des données pour l'entraînement"""
        # Séparation des features et target
        X = df.drop([ID_COLUMN, TARGET_COLUMN], axis=1)
        y = df[TARGET_COLUMN]
        
        # Division train/test
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=MODEL_CONFIG['test_size'], 
            random_state=MODEL_CONFIG['random_state'], stratify=y
        )
        
        # Application de SMOTE pour équilibrer les classes
        smote = SMOTE(random_state=MODEL_CONFIG['random_state'])
        self.X_train, self.y_train = smote.fit_resample(self.X_train, self.y_train)
        
        print(f"✅ Données préparées: Train {self.X_train.shape}, Test {self.X_test.shape}")
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def define_models(self):
        """Définition des modèles à entraîner"""
        self.models = {
            'random_forest': {
                'model': RandomForestClassifier(random_state=MODEL_CONFIG['random_state']),
                'params': {
                    'n_estimators': [100, 200],
                    'max_depth': [10, 20, None],
                    'min_samples_split': [2, 5]
                }
            },
            'xgboost': {
                'model': XGBClassifier(random_state=MODEL_CONFIG['random_state']),
                'params': {
                    'n_estimators': [100, 200],
                    'max_depth': [6, 10],
                    'learning_rate': [0.1, 0.2]
                }
            },
            'svm': {
                'model': SVC(random_state=MODEL_CONFIG['random_state'], probability=True),
                'params': {
                    'C': [0.1, 1, 10],
                    'kernel': ['rbf', 'linear']
                }
            }
        }
    
    def train_model(self, model_name, model_config):
        """Entraînement d'un modèle avec GridSearchCV"""
        print(f"🔄 Entraînement du modèle {model_name}...")
        
        # Grid Search avec validation croisée
        grid_search = GridSearchCV(
            model_config['model'], 
            model_config['params'],
            cv=MODEL_CONFIG['cv_folds'], 
            scoring='roc_auc',
            n_jobs=-1
        )
        
        grid_search.fit(self.X_train, self.y_train)
        
        # Évaluation sur le set de test
        y_pred = grid_search.predict(self.X_test)
        y_pred_proba = grid_search.predict_proba(self.X_test)[:, 1]
        
        # Calcul des métriques
        auc_score = roc_auc_score(self.y_test, y_pred_proba)
        
        # Sauvegarde du modèle
        model_path = TRAINED_MODELS_DIR / f"{model_name}_model.pkl"
        joblib.dump(grid_search.best_estimator_, model_path)
        
        # Mise à jour du meilleur modèle
        if auc_score > self.best_score:
            self.best_score = auc_score
            self.best_model = grid_search.best_estimator_
            joblib.dump(self.best_model, TRAINED_MODELS_DIR / "best_classifier.pkl")
        
        print(f"✅ {model_name} - AUC Score: {auc_score:.4f}")
        return grid_search.best_estimator_, auc_score
    
    def evaluate_model(self, model, model_name):
        """Évaluation détaillée d'un modèle"""
        y_pred = model.predict(self.X_test)
        y_pred_proba = model.predict_proba(self.X_test)[:, 1]
        
        # Rapport de classification
        report = classification_report(self.y_test, y_pred, output_dict=True)
        
        # Matrice de confusion
        cm = confusion_matrix(self.y_test, y_pred)
        
        # AUC Score
        auc = roc_auc_score(self.y_test, y_pred_proba)
        
        # Sauvegarde des résultats
        results = {
            'model_name': model_name,
            'auc_score': auc,
            'classification_report': report,
            'confusion_matrix': cm.tolist()
        }
        
        return results
    
    def get_feature_importance(self):
        """Extraction de l'importance des variables"""
        if hasattr(self.best_model, 'feature_importances_'):
            feature_names = self.X_train.columns
            importance = self.best_model.feature_importances_
            
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importance
            }).sort_values('importance', ascending=False)
            
            # Sauvegarde
            importance_df.to_csv(OUTPUTS_DIR / "feature_importance.csv", index=False)
            
            return importance_df
        
        return None
    
    def train_all_models(self):
        """Entraînement de tous les modèles"""
        print("🔄 Début de l'entraînement des modèles...")
        
        # Chargement des données
        df = self.load_processed_data()
        if df is None:
            return None
        
        # Préparation des données
        self.prepare_data(df)
        
        # Définition des modèles
        self.define_models()
        
        # Entraînement de chaque modèle
        results = {}
        for model_name, model_config in self.models.items():
            model, score = self.train_model(model_name, model_config)
            results[model_name] = self.evaluate_model(model, model_name)
        
        # Extraction de l'importance des variables
        importance_df = self.get_feature_importance()
        
        print(f"✅ Entraînement terminé! Meilleur modèle: AUC = {self.best_score:.4f}")
        
        return results, importance_df

def run_model_training():
    """Fonction principale pour l'entraînement des modèles"""
    trainer = ModelTrainer()
    return trainer.train_all_models()

if __name__ == "__main__":
    run_model_training()