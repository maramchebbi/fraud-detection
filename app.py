import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow import keras
import pickle

st.set_page_config(
    page_title="Détection de Fraude LSTM - Maram Chebbi",
    page_icon="🔍",
    layout="wide"
)

@st.cache_resource
def load_model():
    model = keras.models.load_model('lstm_fraud_model.h5')
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('metadata.pkl', 'rb') as f:
        metadata = pickle.load(f)
    return model, scaler, metadata

try:
    model, scaler, metadata = load_model()
    model_loaded = True
except:
    model_loaded = False

st.markdown("""
<style>
    .stButton>button {
        background: linear-gradient(135deg, #1A367E 0%, #4A8FE7 100%);
        color: white;
        font-weight: bold;
        border: none;
        padding: 12px 30px;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

st.title("🔍 Détection de Fraude LSTM")
st.markdown("### Système intelligent de détection de fraude d'assurance")
st.markdown("**Développé par** : Maram Chebbi | ESPRIT & IRA Le Mans")
st.markdown("---")

if not model_loaded:
    st.error("⚠️ Modèle non chargé. Veuillez uploader les fichiers requis.")
    st.stop()

st.sidebar.header("📊 Informations")
st.sidebar.metric("Features", len(metadata['feature_names']))
st.sidebar.markdown("### 🎯 Performance")
st.sidebar.metric("Accuracy", "89%")
st.sidebar.metric("ROC-AUC", "0.94")

st.subheader("📝 Informations de la Transaction")

st.info("💡 **Note**: Entrez les valeurs normalisées des features. Utilisez des valeurs entre -3 et 3 pour des transactions typiques.")

feature_names = metadata['feature_names'][:10]

feature_labels = {
    'Time': '⏰ Temps (secondes depuis première transaction)',
    'V1': '🔢 Feature V1 (Composante PCA 1)',
    'V2': '🔢 Feature V2 (Composante PCA 2)',
    'V3': '🔢 Feature V3 (Composante PCA 3)',
    'V4': '🔢 Feature V4 (Composante PCA 4)',
    'V5': '🔢 Feature V5 (Composante PCA 5)',
    'V6': '🔢 Feature V6 (Composante PCA 6)',
    'V7': '🔢 Feature V7 (Composante PCA 7)',
    'V8': '🔢 Feature V8 (Composante PCA 8)',
    'V9': '🔢 Feature V9 (Composante PCA 9)',
    'Amount': '💰 Montant de la Transaction (€)'
}

feature_descriptions = {
    'Time': 'Temps écoulé en secondes depuis la première transaction du dataset',
    'Amount': 'Montant de la transaction en euros',
}

col1, col2 = st.columns(2)

inputs = {}

with col1:
    st.markdown("#### ⏰ Informations Temporelles")
    if 'Time' in feature_names:
        inputs['Time'] = st.number_input(
            '⏰ Temps (secondes)',
            min_value=0.0,
            max_value=200000.0,
            value=0.0,
            step=1000.0,
            help='Temps écoulé depuis la première transaction'
        )
    
    st.markdown("#### 💰 Montant")
    if 'Amount' in feature_names:
        inputs['Amount'] = st.number_input(
            '💰 Montant (€)',
            min_value=0.0,
            max_value=10000.0,
            value=100.0,
            step=10.0,
            help='Montant de la transaction en euros'
        )

with col2:
    st.markdown("#### 🔢 Features Transformées (PCA)")
    st.caption("Valeurs normalisées issues de l'Analyse en Composantes Principales")
    
    for feature in feature_names:
        if feature not in ['Time', 'Amount']:
            inputs[feature] = st.number_input(
                f'{feature}',
                min_value=-5.0,
                max_value=5.0,
                value=0.0,
                step=0.1,
                help='Composante PCA normalisée'
            )

st.markdown("---")

with st.expander("ℹ️ Qu'est-ce que les features V1-V28 ?"):
    st.markdown("""
    ### Features Anonymisées
    
    Pour des raisons de **confidentialité**, les features originales ont été transformées 
    via une **Analyse en Composantes Principales (PCA)**.
    
    **Ce que vous devez savoir** :
    - **V1 à V28** : Composantes principales issues de la transformation PCA
    - **Time** : Temps en secondes depuis la première transaction
    - **Amount** : Montant réel de la transaction en euros
    
    **Valeurs typiques** :
    - Features V1-V28 : Entre -3 et +3 pour 99% des transactions
    - Time : 0 à 172,800 (48 heures)
    - Amount : 0 à 25,000€ (moyenne ~88€)
    
    **Pour tester** :
    - Transaction normale : Laissez toutes les V à 0, montant = 100€
    - Transaction suspecte : Mettez quelques V à ±3, montant élevé
    """)

st.markdown("---")

with st.expander("🎯 Exemples de Transactions"):
    st.markdown("""
    ### Transaction NORMALE ✅
    - Time: 5000
    - V1 à V28: 0
    - Amount: 50€
    
    ### Transaction SUSPECTE 🚨
    - Time: 80000
    - V1: 2.5, V2: -3.1, V3: 1.8
    - V4-V28: 0
    - Amount: 5000€
    
    ### Petite Transaction LÉGITIME ✅
    - Time: 1000
    - Toutes V: 0
    - Amount: 10€
    """)

if st.button("🔍 Analyser la réclamation", use_container_width=True):
    with st.spinner("Analyse en cours..."):
        features_list = [inputs.get(feat, 0) for feat in metadata['feature_names']]
        features_array = np.array(features_list).reshape(1, -1)
        features_scaled = scaler.transform(features_array)
        features_lstm = features_scaled.reshape((1, 1, -1))
        
        proba = float(model.predict(features_lstm, verbose=0)[0][0])
        
        st.markdown("---")
        st.subheader("📈 Résultat de l'analyse")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if proba > 0.5:
                st.error("⚠️ FRAUDE DÉTECTÉE")
            else:
                st.success("✅ LÉGITIME")
        
        with col2:
            st.metric("Probabilité de fraude", f"{proba*100:.2f}%")
        
        with col3:
            if proba > 0.7:
                st.error("Risque : Élevé")
            elif proba > 0.3:
                st.warning("Risque : Moyen")
            else:
                st.success("Risque : Faible")
        
        st.progress(proba)
        
        if proba > 0.5:
            st.warning("🚨 Cette réclamation présente des caractéristiques suspectes. Investigation recommandée.")
        else:
            st.info("✅ Cette réclamation semble légitime selon le modèle.")

st.markdown("---")
st.markdown("**💡 Note** : Ce système utilise un réseau LSTM entraîné sur des données réelles d'assurance.")
st.markdown("**📧 Contact** : chebbimaram0@gmail.com | [LinkedIn](https://linkedin.com/in/maramchebbi) | [GitHub](https://github.com/maramchebbi)")
