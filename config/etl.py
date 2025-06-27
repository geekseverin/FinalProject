"""
Pipeline ETL complet pour le projet de prédiction d'abandon scolaire
Execute toutes les étapes de traitement des données et d'entraînement des modèles
"""

import sys
import os
from pathlib import Path
import time
import pandas as pd

# Ajout du chemin pour l'importation des modules
sys.path.append(str(Path(__file__).parent))

# Importation des modules du projet
from src.data_preprocessing import run_preprocessing
from src.model_training import run_model_training  
from src.clustering_analysis import run_clustering
from src.association_rules import run_association_rules
from config.config import *

def create_sample_data():
    """Création des données d'exemple si elles n'existent pas"""
    data_path = RAW_DATA_DIR / "student_data.csv"
    
    if not data_path.exists():
        print("📋 Création des données d'exemple...")
        
        # Données d'exemple basées sur votre exemple
        sample_data = {
            'student_id': range(1, 101),
            'age': np.random.randint(18, 35, 100),
            'gender': np.random.choice(['Male', 'Female'], 100),
            'region': np.random.choice(['Notse', 'Tsevie', 'Lome', 'Dapaong', 'Sokode'], 100),
            'parent_education': np.random.choice(['None', 'Primary', 'Secondary', 'Higher'], 100),
            'average_grade': np.random.uniform(8, 20, 100).round(2),
            'absenteeism_rate': np.random.uniform(0, 30, 100).round(2),
            'assignments_submitted': np.random.uniform(40, 100, 100).round(2),
            'moodle_hours': np.random.uniform(0, 25, 100).round(2),
            'forum_posts': np.random.randint(0, 10, 100),
            'satisfaction_score': np.random.uniform(1, 10, 100).round(1),
            'dropout': np.random.choice(['Yes', 'No'], 100, p=[0.3, 0.7])
        }
        
        df = pd.DataFrame(sample_data)
        df.to_csv(data_path, index=False)
        print(f"✅ Données d'exemple créées: {data_path}")
        
        return df
    else:
        print(f"✅ Données existantes trouvées: {data_path}")
        return pd.read_csv(data_path)

def setup_directories():
    """Création de tous les dossiers nécessaires"""
    print("📁 Configuration des dossiers...")
    
    directories = [
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR, 
        TRAINED_MODELS_DIR,
        CLUSTERING_DIR,
        ASSOCIATION_RULES_DIR,
        OUTPUTS_DIR,
        OUTPUTS_DIR / "reports",
        OUTPUTS_DIR / "plots",
        OUTPUTS_DIR / "logs"
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
    
    print("✅ Tous les dossiers sont configurés")

def log_step(step_name, start_time=None):
    """Logger pour suivre les étapes"""
    if start_time is None:
        print(f"\n{'='*50}")
        print(f"🚀 DÉBUT: {step_name}")
        print(f"{'='*50}")
        return time.time()
    else:
        duration = time.time() - start_time
        print(f"✅ TERMINÉ: {step_name} ({duration:.2f}s)")
        print(f"{'='*50}")
        return duration

def run_etl_pipeline():
    """Exécution du pipeline ETL complet"""
    print("""
    ████████╗ ██████╗  ██████╗  ██████╗ 
    ╚══██╔══╝██╔═══██╗██╔════╝ ██╔═══██╗
       ██║   ██║   ██║██║  ███╗██║   ██║
       ██║   ██║   ██║██║   ██║██║   ██║
       ██║   ╚██████╔╝╚██████╔╝╚██████╔╝
       ╚═╝    ╚═════╝  ╚═════╝  ╚═════╝ 
    
    🎓 PIPELINE DE PRÉDICTION D'ABANDON SCOLAIRE 🎓
    """)
    
    total_start = time.time()
    results = {}
    
    try:
        # 1. Configuration initiale
        step_start = log_step("Configuration et vérification des données")
        setup_directories()
        sample_df = create_sample_data()
        log_step("Configuration et vérification des données", step_start)
        
        # 2. Preprocessing des données
        step_start = log_step("Preprocessing des données")
        preprocessing_result = run_preprocessing()
        if preprocessing_result is not None:
            results['preprocessing'] = True
            log_step("Preprocessing des données", step_start)
        else:
            raise Exception("Échec du preprocessing")
        
        # 3. Entraînement des modèles
        step_start = log_step("Entraînement des modèles de classification")
        training_results = run_model_training()
        if training_results is not None:
            results['model_training'] = training_results
            log_step("Entraînement des modèles de classification", step_start)
        else:
            print("⚠️  Échec de l'entraînement des modèles")
        
        # 4. Analyse de clustering
        step_start = log_step("Analyse de clustering")
        clustering_results = run_clustering()
        if clustering_results is not None:
            results['clustering'] = clustering_results
            log_step("Analyse de clustering", step_start)
        else:
            print("⚠️  Échec de l'analyse de clustering")
        
        # 5. Règles d'association
        step_start = log_step("Extraction des règles d'association")
        association_results = run_association_rules()
        if association_results is not None:
            results['association_rules'] = association_results
            log_step("Extraction des règles d'association", step_start)
        else:
            print("⚠️  Échec de l'extraction des règles d'association")
        
        # 6. Rapport final
        total_duration = time.time() - total_start
        
        print(f"""
        
        🎉 PIPELINE ETL TERMINÉ AVEC SUCCÈS! 🎉
        
        ⏱️  Durée totale: {total_duration:.2f} secondes
        
        📊 RÉSULTATS:
        ✅ Preprocessing: Terminé
        ✅ Modèles entraînés: {len(results.get('model_training', [{}])[0]) if results.get('model_training') else 0}
        ✅ Clusters identifiés: {results.get('clustering', {}).get('best_n_clusters', 'N/A')}
        ✅ Règles d'association: {len(results.get('association_rules', [])) if results.get('association_rules') else 0}