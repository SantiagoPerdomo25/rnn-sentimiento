"""App Streamlit: escribe una reseña de película (en inglés) y obtén su sentimiento."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import streamlit as st

from src.predict import load_model, predict_texts

st.set_page_config(page_title="Análisis de sentimiento", page_icon="🎬")


@st.cache_resource
def get_model():
    return load_model()


EJEMPLOS = {
    "Positiva": "An absolutely wonderful film. The acting was superb and the story kept me hooked until the end.",
    "Negativa": "A complete waste of time. The plot made no sense, the acting was terrible and I almost fell asleep.",
    "Mixta": "The visuals were beautiful, but the story was slow and the ending felt rushed.",
}

st.title("🎬 Análisis de sentimiento con LSTM/GRU")
st.write("Escribe una reseña de película **en inglés** (el modelo se entrenó con reseñas de IMDB) "
         "y la red recurrente dirá si es positiva o negativa.")

try:
    model = get_model()
except Exception as e:
    st.error("No se pudo cargar el modelo. Ejecuta `python -m src.train` o copia "
             "`sentimiento_rnn.keras` a la carpeta `models/`.")
    st.exception(e)
    st.stop()

tab1, tab2 = st.tabs(["Una reseña", "Varias reseñas"])

with tab1:
    ejemplo = st.selectbox("Cargar un ejemplo (opcional)", ["—"] + list(EJEMPLOS))
    texto = st.text_area("Reseña", value=EJEMPLOS.get(ejemplo, ""), height=160,
                         placeholder="Write your movie review here...")
    if st.button("Analizar", type="primary"):
        if not texto.strip():
            st.warning("Escribe un texto primero.")
        else:
            r = predict_texts(model, [texto])[0]
            icono = "😊" if r["label"] == "positivo" else "😞"
            st.subheader(f"{icono} Sentimiento: **{r['label'].upper()}**")
            st.metric("Confianza", f"{r['confidence']:.2%}")
            st.progress(r["confidence"])
            st.caption(f"Probabilidad de ser positiva: {r['prob_positive']:.2%}")
            if r["confidence"] < 0.65:
                st.warning("Confianza baja: el texto podría ser ambiguo o neutro.")

with tab2:
    st.write("Escribe una reseña por línea.")
    bloque = st.text_area("Reseñas", height=200, key="multi")
    if st.button("Analizar todas"):
        lineas = [l.strip() for l in bloque.splitlines() if l.strip()]
        if not lineas:
            st.warning("Escribe al menos una reseña.")
        else:
            res = predict_texts(model, lineas)
            st.dataframe(pd.DataFrame({
                "Reseña": [l[:90] + ("..." if len(l) > 90 else "") for l in lineas],
                "Sentimiento": [r["label"] for r in res],
                "Confianza": [f"{r['confidence']:.1%}" for r in res],
            }), use_container_width=True)
