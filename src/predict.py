"""Carga del modelo y predicción de sentimiento (módulo + CLI)."""
import argparse
from pathlib import Path

import tensorflow as tf

from src.data import CLASS_NAMES, clean_text

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "sentimiento_rnn.keras"


def load_model(model_path=MODEL_PATH):
    return tf.keras.models.load_model(model_path)


def predict_texts(model, texts):
    """Devuelve una lista de dicts con etiqueta, confianza y P(positivo) por cada texto."""
    cleaned = [clean_text(t) for t in texts]
    probs = model.predict(tf.constant(cleaned), verbose=0).ravel()
    out = []
    for p in probs:
        p = float(p)
        label = CLASS_NAMES[int(p >= 0.5)]
        out.append({"label": label, "confidence": p if p >= 0.5 else 1 - p, "prob_positive": p})
    return out


def predict_text(model, text):
    return predict_texts(model, [text])[0]


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Predice el sentimiento de una reseña (en inglés).")
    ap.add_argument("text", help="Texto de la reseña entre comillas")
    args = ap.parse_args()
    r = predict_text(load_model(), args.text)
    print(f"Sentimiento: {r['label']} (confianza {r['confidence']:.2%}, P(positivo)={r['prob_positive']:.3f})")
