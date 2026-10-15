"""Métricas, gráficas y tabla de predicciones."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score, recall_score,
                             roc_auc_score, roc_curve)

from src.data import CLASS_NAMES


def plot_history(history, path=None):
    h = history.history if hasattr(history, "history") else history
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(h["accuracy"], label="Entrenamiento")
    ax[0].plot(h["val_accuracy"], label="Validación")
    ax[0].set(title="Accuracy", xlabel="Época", ylabel="Accuracy")
    ax[0].legend()
    ax[1].plot(h["loss"], label="Entrenamiento")
    ax[1].plot(h["val_loss"], label="Validación")
    ax[1].set(title="Pérdida (loss)", xlabel="Época", ylabel="Loss")
    ax[1].legend()
    plt.tight_layout()
    if path:
        fig.savefig(path, dpi=150)
    return fig


def get_predictions(model, ds):
    """Devuelve (y_true, probabilidades, y_pred) en un dataset sin barajar."""
    y_true = np.concatenate([y.numpy() for _, y in ds]).astype(int)
    probs = model.predict(ds, verbose=0).ravel()
    return y_true, probs, (probs >= 0.5).astype(int)


def metrics_dict(y_true, probs, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, probs),
    }


def text_report(y_true, y_pred):
    return classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=3)


def plot_confusion_matrix(y_true, y_pred, path=None):
    fig, ax = plt.subplots(figsize=(5, 5))
    ConfusionMatrixDisplay(confusion_matrix(y_true, y_pred), display_labels=CLASS_NAMES).plot(
        ax=ax, cmap="Blues")
    ax.set_title("Matriz de confusión (conjunto de prueba)")
    plt.tight_layout()
    if path:
        fig.savefig(path, dpi=150)
    return fig


def plot_roc(y_true, probs, path=None):
    fpr, tpr, _ = roc_curve(y_true, probs)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, label=f"AUC = {roc_auc_score(y_true, probs):.3f}")
    ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.set(title="Curva ROC", xlabel="Tasa de falsos positivos", ylabel="Tasa de verdaderos positivos")
    ax.legend()
    plt.tight_layout()
    if path:
        fig.savefig(path, dpi=150)
    return fig


def prediction_table(model, df, n=12, text_col="text_clean", seed=0):
    """Predicciones sobre n reseñas aleatorias del conjunto indicado."""
    sample = df.sample(n=n, random_state=seed).reset_index(drop=True)
    probs = model.predict(tf.constant(sample[text_col].tolist()), verbose=0).ravel()
    pred = (probs >= 0.5).astype(int)
    return pd.DataFrame({
        "reseña": sample[text_col].str.slice(0, 110) + "...",
        "real": sample["label"].map(dict(enumerate(CLASS_NAMES))),
        "predicción": [CLASS_NAMES[p] for p in pred],
        "prob_positiva": probs.round(3),
        "confianza": np.where(pred == 1, probs, 1 - probs).round(3),
        "acierto": np.where(pred == sample["label"].values, "✅", "❌"),
    })
