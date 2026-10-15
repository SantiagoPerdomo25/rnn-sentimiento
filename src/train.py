"""Entrenamiento completo: python -m src.train --cell lstm --epochs 8"""
import argparse
import json
from pathlib import Path

import tensorflow as tf

from src import data, evaluate, model as model_lib

ROOT = Path(__file__).resolve().parents[1]


def main(cell, epochs, batch_size):
    tf.keras.utils.set_random_seed(data.SEED)
    (ROOT / "models").mkdir(exist_ok=True)
    (ROOT / "reports").mkdir(exist_ok=True)

    df = data.load_dataframe(data.download_dataset(ROOT / "data"))
    df = df.drop_duplicates(subset="text").reset_index(drop=True)
    df["text_clean"] = df["text"].map(data.clean_text)
    splits = data.split_data(df)
    print({k: len(v) for k, v in splits.items()})

    ds = {k: data.make_dataset(v["text_clean"], v["label"], batch_size, training=(k == "train"))
          for k, v in splits.items()}

    vectorizer = model_lib.build_vectorizer(splits["train"]["text_clean"])
    model = model_lib.build_model(vectorizer, cell)
    model.summary()

    callbacks = [tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=2,
                                                  restore_best_weights=True)]
    history = model.fit(ds["train"], validation_data=ds["val"], epochs=epochs, callbacks=callbacks)

    model.save(ROOT / "models" / "sentimiento_rnn.keras")
    (ROOT / "reports" / "history.json").write_text(json.dumps(history.history), encoding="utf-8")

    evaluate.plot_history(history, ROOT / "reports" / "training_curves.png")
    y_true, probs, y_pred = evaluate.get_predictions(model, ds["test"])
    m = evaluate.metrics_dict(y_true, probs, y_pred)
    (ROOT / "reports" / "metrics.json").write_text(json.dumps(m, indent=2), encoding="utf-8")
    report = evaluate.text_report(y_true, y_pred)
    (ROOT / "reports" / "classification_report.txt").write_text(report, encoding="utf-8")
    print(report, m)
    evaluate.plot_confusion_matrix(y_true, y_pred, ROOT / "reports" / "confusion_matrix.png")
    evaluate.plot_roc(y_true, probs, ROOT / "reports" / "roc_curve.png")
    evaluate.prediction_table(model, splits["test"], n=12).to_csv(
        ROOT / "reports" / "predictions.csv", index=False, encoding="utf-8")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", default="lstm", choices=["lstm", "gru", "simple"])
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch_size", type=int, default=64)
    a = ap.parse_args()
    main(a.cell, a.epochs, a.batch_size)
