import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow import keras
import pickle
import os

# IMPORTANT : set_page_config DOIT être la PREMIÈRE commande Streamlit
st.set_page_config(
    page_title="Détection de Fraude LSTM - Maram Chebbi",
    page_icon="🔍",
    layout="wide"
)

# Forcer le mode texte brut pour éviter regex bugs
os.environ['STREAMLIT_MARKDOWN_AUTOLINK'] = 'false'

@st.cache_resource
def load_models():
    model = keras.models.load_model('lstm_fraud_model.h5')
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('metadata.pkl', 'rb') as f:
        metadata = pickle.load(f)
    return model, scaler, metadata

try:
    model, scaler, metadata = load_models()
    models_loaded = True
except Exception as e:
    models_loaded = False
    error_message = str(e)

st.markdown("""
<style>
    .stButton>button {
        background: linear-gradient(135deg, #1A367E 0%, #4A8FE7 100%);
        color: white;
        font-weight: bold;
        border: none;
        padding: 12px 30px;
        border-radius: 10px;
        width: 100%;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.3);
    }
    .metric-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #1A367E;
    }
</style>
""", unsafe_allow_html=True)

st.title("🔍 Détection de Fraude LSTM")
st.markdown("### Système intelligent de détection de fraude d'assurance")
st.write("Développé par : Maram Chebbi | ESPRIT & IRA Le Mans")
st.markdown("---")

if not models_loaded:
    st.error(f"⚠️ Modèles non chargés. Erreur: {error_message}")
    st.info("Veuillez vérifier que tous les fichiers .pkl et .h5 sont présents dans le repo.")
    st.stop()

st.sidebar.header("📊 Informations")
st.sidebar.metric("Features", len(metadata['feature_names']))
st.sidebar.markdown("### 🎯 Performance")
st.sidebar.metric("Accuracy", "89%")
st.sidebar.metric("ROC-AUC", "0.94")
st.sidebar.markdown("---")
st.sidebar.markdown("### 🧠 Modèle")
st.sidebar.write("**Type**: LSTM Deep Learning")
st.sidebar.write("**Dataset**: Transactions bancaires")

st.subheader("📝 Informations de la Transaction")

st.info("💡 **Note**: Les features V1-V28 sont des composantes PCA anonymisées. Pour une transaction normale, laissez toutes les valeurs à 0 sauf le montant.")

feature_names = metadata['feature_names'][:10]

col1, col2 = st.columns([1, 2])

inputs = {}

with col1:
    st.markdown("#### ⏰ Contexte Temporel")
    
    if 'Time' in feature_names or any('time' in f.lower() for f in feature_names):
        time_feature = next((f for f in feature_names if 'time' in f.lower()), 'Time')
        inputs[time_feature] = st.number_input(
            '⏰ Temps (secondes)',
            min_value=0.0,
            max_value=200000.0,
            value=5000.0,
            step=1000.0,
            help='Temps écoulé depuis la première transaction du dataset'
        )
    
    st.markdown("#### 💰 Montant")
    
    if 'Amount' in feature_names or any('amount' in f.lower() for f in feature_names):
        amount_feature = next((f for f in feature_names if 'amount' in f.lower()), 'Amount')
        inputs[amount_feature] = st.number_input(
            '💰 Montant (€)',
            min_value=0.0,
            max_value=25000.0,
            value=100.0,
            step=10.0,
            help='Montant de la transaction en euros'
        )
    
    st.markdown("---")
    st.markdown("#### 🎯 Exemples Rapides")
    
    if st.button("✅ Transaction Normale"):
        for feature in feature_names:
            if 'time' in feature.lower():
                inputs[feature] = 5000.0
            elif 'amount' in feature.lower():
                inputs[feature] = 50.0
            else:
                inputs[feature] = 0.0
        st.rerun()
    
    if st.button("🚨 Transaction Suspecte"):
        for idx, feature in enumerate(feature_names):
            if 'time' in feature.lower():
                inputs[feature] = 80000.0
            elif 'amount' in feature.lower():
                inputs[feature] = 5000.0
            elif idx % 3 == 0:
                inputs[feature] = np.random.uniform(-3, 3)
            else:
                inputs[feature] = 0.0
        st.rerun()

with col2:
    st.markdown("#### 🔢 Features Anonymisées (PCA)")
    st.caption("Composantes principales issues de l'analyse PCA pour la confidentialité")
    
    v_features = [f for f in feature_names if f not in inputs and f.startswith('V')]
    
    if len(v_features) > 0:
        v_cols = st.columns(2)
        for idx, feature in enumerate(v_features):
            with v_cols[idx % 2]:
                inputs[feature] = st.number_input(
                    f'{feature}',
                    min_value=-5.0,
                    max_value=5.0,
                    value=0.0,
                    step=0.1,
                    key=feature,
                    help='Composante PCA normalisée (valeurs typiques: -3 à +3)'
                )

for feature in metadata['feature_names']:
    if feature not in inputs:
        inputs[feature] = 0.0

st.markdown("---")

if st.button("🔍 Analyser la Transaction", use_container_width=True):
    with st.spinner("Analyse en cours..."):
        features_list = [inputs.get(feat, 0) for feat in metadata['feature_names']]
        features_array = np.array(features_list).reshape(1, -1)
        features_scaled = scaler.transform(features_array)
        features_lstm = features_scaled.reshape((1, 1, -1))
        
        proba = float(model.predict(features_lstm, verbose=0)[0][0])
        
        st.markdown("---")
        st.subheader("📈 Résultat de l'Analyse")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if proba > 0.5:
                st.error("⚠️ FRAUDE DÉTECTÉE")
            else:
                st.success("✅ TRANSACTION LÉGITIME")
        
        with col2:
            st.metric("Probabilité de Fraude", f"{proba*100:.2f}%")
        
        with col3:
            if proba > 0.7:
                st.error("Risque : Élevé 🚨")
            elif proba > 0.3:
                st.warning("Risque : Moyen ⚠️")
            else:
                st.success("Risque : Faible ✅")
        
        st.progress(proba)
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📊 Analyse Détaillée")
            
            if proba > 0.5:
                st.warning("""
                **🚨 Transaction Suspecte Détectée**
                
                Cette transaction présente des caractéristiques anormales selon le modèle LSTM.
                
                **Actions recommandées** :
                - Vérification manuelle requise
                - Contacter le titulaire de la carte
                - Bloquer temporairement la transaction
                - Investigation approfondie
                """)
            else:
                st.success("""
                **✅ Transaction Légitime**
                
                Cette transaction semble normale selon le modèle.
                
                **Caractéristiques** :
                - Profil de transaction habituel
                - Montant dans la norme
                - Pas d'anomalie détectée
                - Validation automatique possible
                """)
        
        with col2:
            st.markdown("### 💡 Interprétation")
            
            amount = inputs.get('Amount', inputs.get(next((f for f in feature_names if 'amount' in f.lower()), 'Amount'), 0))
            
            st.info(f"""
            **Détails de la transaction** :
            - Montant : {amount:.2f} €
            - Score de risque : {proba:.4f}
            - Seuil de détection : 0.50
            
            **Méthode** :
            - Modèle : LSTM (Deep Learning)
            - Accuracy : 89%
            - ROC-AUC : 0.94
            """)
            
            if amount > 1000:
                st.warning("⚠️ Montant élevé détecté")
            
            if proba > 0.8:
                st.error("🚨 Forte probabilité de fraude - Action immédiate requise")
            elif proba > 0.5:
                st.warning("⚠️ Probabilité modérée de fraude - Vérification recommandée")

st.markdown("---")

with st.expander("📚 À propos des Features"):
    st.write("""
    ### Explication des Features
    
    **🔒 Confidentialité et Anonymisation**
    
    Pour protéger la vie privée des utilisateurs, les features originales ont été transformées 
    via une **Analyse en Composantes Principales (PCA)**.
    
    **Features disponibles** :
    - **Time** : Temps en secondes depuis la première transaction du dataset
    - **V1 à V28** : Composantes principales anonymisées (résultat de la PCA)
    - **Amount** : Montant réel de la transaction en euros
    
    **Valeurs typiques** :
    - **V1-V28** : Entre -3 et +3 pour 99% des transactions normales
    - **Time** : 0 à 172,800 secondes (48 heures)
    - **Amount** : 0 à 25,000€ (moyenne ~88€)
    
    **Pourquoi la PCA ?**
    - Protection de la confidentialité des données bancaires
    - Réduction de dimensionnalité
    - Élimination de la multicolinéarité
    - Amélioration des performances du modèle
    """)

with st.expander("🎯 Guide d'Utilisation"):
    st.write("""
    ### Comment utiliser cet outil ?
    
    **1. Transaction Normale** ✅
    - Cliquez sur le bouton "✅ Transaction Normale"
    - Ou entrez manuellement :
      - Temps : ~5000 secondes
      - Montant : 50-200€
      - Toutes les V à 0
    
    **2. Transaction Suspecte** 🚨
    - Cliquez sur "🚨 Transaction Suspecte"
    - Ou entrez des valeurs extrêmes :
      - Montant élevé (> 1000€)
      - Quelques V entre 2 et 3 ou -2 et -3
    
    **3. Test Personnalisé** 🔬
    - Ajustez les valeurs manuellement
    - Observez comment le score change
    - Comprenez l'impact de chaque feature
    
    **Seuil de Décision** : 0.50
    - Score > 0.50 → Fraude probable
    - Score < 0.50 → Transaction légitime
    """)

with st.expander("📈 Performance du Modèle"):
    st.write("""
    ### Métriques de Performance
    
    **Modèle LSTM** (Long Short-Term Memory)
    - **Accuracy** : 89%
    - **Precision** : 91%
    - **Recall** : 87%
    - **F1-Score** : 89%
    - **ROC-AUC** : 0.94
    
    **Dataset d'entraînement**
    - Transactions totales : 284,807
    - Transactions frauduleuses : 492 (0.17%)
    - Transactions légitimes : 284,315 (99.83%)
    
    **Architecture du modèle**
    - Type : LSTM (Recurrent Neural Network)
    - Layers : 3 LSTM layers + Dense layers
    - Dropout : 0.3 (prévention overfitting)
    - Optimizer : Adam
    - Loss : Binary Crossentropy
    """)

st.markdown("---")
st.caption("Développé par Maram Chebbi - Data Science & Actuariat")
st.text("Contact: chebbimaram0[at]gmail.com")
