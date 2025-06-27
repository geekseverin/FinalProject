"""
Configuration centralisée pour le projet de prédiction d'abandon scolaire
"""

import os
from pathlib import Path

# Chemins des dossiers
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"

# Chemins des sous-dossiers de modèles
TRAINED_MODELS_DIR = MODELS_DIR / "trained_models"
CLUSTERING_DIR = MODELS_DIR / "clustering"
ASSOCIATION_RULES_DIR = MODELS_DIR / "association_rules"

# Création des dossiers s'ils n'existent pas
for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, 
                 TRAINED_MODELS_DIR, CLUSTERING_DIR, ASSOCIATION_RULES_DIR, OUTPUTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Configuration des modèles
MODEL_CONFIG = {
    'random_state': 42,
    'test_size': 0.2,
    'cv_folds': 5,
    'n_clusters_range': range(2, 8),
    'min_support': 0.1,
    'min_confidence': 0.5
}

# Configuration des colonnes
FEATURE_COLUMNS = [
    'age', 'gender', 'region', 'parent_education', 'average_grade',
    'absenteeism_rate', 'assignments_submitted', 'moodle_hours',
    'forum_posts', 'satisfaction_score'
]

TARGET_COLUMN = 'dropout'
ID_COLUMN = 'student_id'

# Configuration Streamlit
STREAMLIT_CONFIG = {
    'page_title': 'Prédiction d\'Abandon Scolaire',
    'page_icon': '🎓',
    'layout': 'wide'
}