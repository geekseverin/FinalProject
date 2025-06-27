"""
Module d'extraction des règles d'association
"""

import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import joblib
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))
from config.config import *

class AssociationRulesAnalyzer:
    def __init__(self):
        self.frequent_itemsets = None
        self.rules = None
        self.recommendations = {}
        
    def load_processed_data(self):
        """Charge les données préprocessées"""
        try:
            data_path = PROCESSED_DATA_DIR / "cleaned_data.csv"
            return pd.read_csv(data_path)
        except Exception as e:
            print(f"Erreur lors du chargement des données: {e}")
            return None
    
    def prepare_transaction_data(self, df):
        """Préparation des données pour l'analyse des règles d'association"""
        print("🔄 Préparation des données transactionnelles...")
        
        # Création des bins pour les variables continues
        df_binned = df.copy()
        
        # Binning des variables continues
        df_binned['age_group'] = pd.cut(df['age'], bins=[0, 25, 30, 100], labels=['Young', 'Middle', 'Senior'])
        df_binned['grade_level'] = pd.cut(df['average_grade'], bins=[0, 10, 15, 20], labels=['Low_Grade', 'Medium_Grade', 'High_Grade'])
        df_binned['absenteeism_level'] = pd.cut(df['absenteeism_rate'], bins=[0, 10, 20, 100], labels=['Low_Absent', 'Medium_Absent', 'High_Absent'])
        df_binned['satisfaction_level'] = pd.cut(df['satisfaction_score'], bins=[0, 3, 6, 10], labels=['Low_Satisfaction', 'Medium_Satisfaction', 'High_Satisfaction'])
        df_binned['forum_activity'] = pd.cut(df['forum_posts'], bins=[0, 2, 5, 100], labels=['Low_Forum', 'Medium_Forum', 'High_Forum'])
        df_binned['moodle_usage'] = pd.cut(df['moodle_hours'], bins=[0, 5, 15, 100], labels=['Low_Moodle', 'Medium_Moodle', 'High_Moodle'])
        
        # Conversion en format transactionnel
        categorical_columns = [
            'gender', 'region', 'parent_education', 'age_group', 
            'grade_level', 'absenteeism_level', 'satisfaction_level',
            'forum_activity', 'moodle_usage', 'dropout'
        ]
        
        # Filtrage des colonnes existantes
        existing_columns = [col for col in categorical_columns if col in df_binned.columns]
        df_categorical = df_binned[existing_columns]
        
        # Conversion des valeurs en chaînes et ajout de préfixes
        transactions = []
        for _, row in df_categorical.iterrows():
            transaction = []
            for col in df_categorical.columns:
                if pd.notna(row[col]):
                    if col == 'dropout':
                        value = 'Dropout_Yes' if row[col] == 1 or row[col] == 'Yes' else 'Dropout_No'
                    else:
                        value = f"{col}_{row[col]}"
                    transaction.append(value)
            transactions.append(transaction)
        
        # Encodage des transactions
        te = TransactionEncoder()
        te_ary = te.fit_transform(transactions)
        df_encoded = pd.DataFrame(te_ary, columns=te.columns_)
        
        return df_encoded, transactions
    
    def extract_frequent_itemsets(self, df_encoded):
        """Extraction des itemsets fréquents"""
        print("🔄 Extraction des itemsets fréquents...")
        
        # Application de l'algorithme Apriori
        self.frequent_itemsets = apriori(
            df_encoded, 
            min_support=MODEL_CONFIG['min_support'], 
            use_colnames=True
        )
        
        if len(self.frequent_itemsets) == 0:
            print("⚠️  Aucun itemset fréquent trouvé. Réduction du seuil de support.")
            self.frequent_itemsets = apriori(
                df_encoded, 
                min_support=0.05, 
                use_colnames=True
            )
        
        print(f"✅ {len(self.frequent_itemsets)} itemsets fréquents trouvés")
        return self.frequent_itemsets
    
    def generate_association_rules(self):
        """Génération des règles d'association"""
        print("🔄 Génération des règles d'association...")
        
        if self.frequent_itemsets is None or len(self.frequent_itemsets) == 0:
            print("❌ Pas d'itemsets fréquents disponibles")
            return None
        
        try:
            # Génération des règles
            self.rules = association_rules(
                self.frequent_itemsets,
                metric="confidence",
                min_threshold=MODEL_CONFIG['min_confidence']
            )
            
            if len(self.rules) == 0:
                print("⚠️  Aucune règle trouvée. Réduction du seuil de confiance.")
                self.rules = association_rules(
                    self.frequent_itemsets,
                    metric="confidence",
                    min_threshold=0.3
                )
            
            # Tri par lift et confiance
            self.rules = self.rules.sort_values(['lift', 'confidence'], ascending=False)
            
            print(f"✅ {len(self.rules)} règles d'association générées")
            return self.rules
            
        except Exception as e:
            print(f"❌ Erreur lors de la génération des règles: {e}")
            return None
    
    def extract_dropout_rules(self):
        """Extraction des règles liées à l'abandon scolaire"""
        if self.rules is None or len(self.rules) == 0:
            return None
        
        # Filtrage des règles liées au dropout
        dropout_rules = self.rules[
            self.rules['consequents'].astype(str).str.contains('Dropout_Yes') |
            self.rules['antecedents'].astype(str).str.contains('Dropout_Yes')
        ]
        
        # Focus sur les règles prédisant l'abandon
        prediction_rules = dropout_rules[
            dropout_rules['consequents'].astype(str).str.contains('Dropout_Yes')
        ]
        
        return prediction_rules
    
    def generate_recommendations(self, dropout_rules):
        """Génération des recommandations basées sur les règles"""
        print("🔄 Génération des recommandations...")
        
        recommendations = {
            'high_risk_patterns': [],
            'intervention_suggestions': [],
            'prevention_strategies': []
        }
        
        if dropout_rules is None or len(dropout_rules) == 0:
            # Recommandations génériques si pas de règles spécifiques
            recommendations['intervention_suggestions'] = [
                "Mettre en place un suivi personnalisé pour les étudiants à risque",
                "Encourager la participation aux forums de discussion",
                "Organiser des sessions de rattrapage pour les étudiants en difficulté",
                "Améliorer l'engagement sur la plateforme Moodle"
            ]
            return recommendations
        
        # Analyse des règles pour générer des recommandations
        for _, rule in dropout_rules.head(10).iterrows():
            antecedents = list(rule['antecedents'])
            confidence = rule['confidence']
            lift = rule['lift']
            
            # Extraction des patterns à risque
            risk_pattern = {
                'conditions': antecedents,
                'dropout_risk': confidence,
                'strength': lift
            }
            recommendations['high_risk_patterns'].append(risk_pattern)
            
            # Génération de suggestions d'intervention
            suggestions = self._generate_interventions_from_pattern(antecedents)
            recommendations['intervention_suggestions'].extend(suggestions)
        
        # Suppression des doublons
        recommendations['intervention_suggestions'] = list(set(recommendations['intervention_suggestions']))
        
        self.recommendations = recommendations
        return recommendations
    
    def _generate_interventions_from_pattern(self, pattern):
        """Génération d'interventions basées sur un pattern"""
        suggestions = []
        
        for condition in pattern:
            if 'Low_Grade' in condition:
                suggestions.append("Proposer du tutorat académique et des sessions de révision")
            elif 'High_Absent' in condition:
                suggestions.append("Mettre en place un système d'alerte précoce pour l'absentéisme")
            elif 'Low_Satisfaction' in condition:
                suggestions.append("Améliorer l'expérience étudiante et recueillir des feedbacks")
            elif 'Low_Forum' in condition:
                suggestions.append("Encourager la participation aux discussions en ligne")
            elif 'Low_Moodle' in condition:
                suggestions.append("Organiser des formations sur l'utilisation de Moodle")
            elif 'None' in condition and 'parent_education' in condition:
                suggestions.append("Offrir un accompagnement renforcé aux étudiants de première génération")
        
        return suggestions
    
    def save_association_results(self):
        """Sauvegarde des résultats d'association"""
        # Sauvegarde des itemsets fréquents
        if self.frequent_itemsets is not None:
            joblib.dump(self.frequent_itemsets, ASSOCIATION_RULES_DIR / "frequent_itemsets.pkl")
        
        # Sauvegarde des règles
        if self.rules is not None:
            joblib.dump(self.rules, ASSOCIATION_RULES_DIR / "rules.pkl")
            self.rules.to_csv(OUTPUTS_DIR / "association_rules.csv", index=False)
        
        # Sauvegarde des recommandations
        if self.recommendations:
            joblib.dump(self.recommendations, ASSOCIATION_RULES_DIR / "recommendations.pkl")
    
    def run_association_analysis(self):
        """Pipeline complet d'analyse des règles d'association"""
        print("🔄 Début de l'analyse des règles d'association...")
        
        # Chargement des données
        df = self.load_processed_data()
        if df is None:
            return None
        
        # Préparation des données transactionnelles
        df_encoded, transactions = self.prepare_transaction_data(df)
        print(f"✅ Données transactionnelles préparées: {df_encoded.shape}")
        
        # Extraction des itemsets fréquents
        frequent_itemsets = self.extract_frequent_itemsets(df_encoded)
        
        # Génération des règles d'association
        rules = self.generate_association_rules()
        
        # Extraction des règles liées au dropout
        dropout_rules = self.extract_dropout_rules()
        
        # Génération des recommandations
        recommendations = self.generate_recommendations(dropout_rules)
        
        # Sauvegarde des résultats
        self.save_association_results()
        
        print("✅ Analyse des règles d'association terminée!")
        
        return {
            'frequent_itemsets': frequent_itemsets,
            'rules': rules,
            'dropout_rules': dropout_rules,
            'recommendations': recommendations,
            'n_transactions': len(transactions)
        }

def run_association_rules():
    """Fonction principale pour l'analyse des règles d'association"""
    analyzer = AssociationRulesAnalyzer()
    return analyzer.run_association_analysis()

if __name__ == "__main__":
    run_association_rules()