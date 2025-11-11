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

st.subheader("📝 Entrez les données de la réclamation")

col1, col2 = st.columns(2)

inputs = {}
feature_list = metadata['feature_names'][:10]

for idx, feature in enumerate(feature_list):
    with col1 if idx % 2 == 0 else col2:
        inputs[feature] = st.number_input(
            feature.replace('_', ' ').title(),
            value=0.0,
            key=feature,
            help=f"Entrez la valeur pour {feature}"
        )

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
