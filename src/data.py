"""Descarga, carga, limpieza y división del dataset IMDB (reseñas de películas)."""
import re
import tarfile
import urllib.request
from pathlib import Path

import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

DATASET_URL = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
SEED = 42
CLASS_NAMES = ["negativo", "positivo"]  # etiqueta 0 / 1

_TAGS = re.compile(r"<[^>]+>")
_URLS = re.compile(r"https?://\S+|www\.\S+")
_SPACES = re.compile(r"\s+")


def _find_imdb(root):
    direct = root / "aclImdb"
    if (direct / "train").is_dir():
        return direct
    return next((p for p in root.rglob("aclImdb") if (p / "train").is_dir()), None)


def download_dataset(cache_dir="data"):
    """Descarga y descomprime IMDB con urllib + tarfile (más fiable que tf.keras.utils.get_file).
    Devuelve la carpeta 'aclImdb' (con train/ y test/)."""
    root = Path(cache_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    found = _find_imdb(root)
    if found:
        return found

    archive = root / "aclImdb_v1.tar.gz"
    if not archive.exists() or archive.stat().st_size < 80_000_000:
        print("Descargando IMDB (~80 MB)...")
        urllib.request.urlretrieve(DATASET_URL, archive)
    print("Descomprimiendo (puede tardar 1-2 minutos)...")
    with tarfile.open(archive, "r:gz") as tar:
        try:
            tar.extractall(root, filter="data")
        except TypeError:          # Python sin soporte del parámetro filter
            tar.extractall(root)

    found = _find_imdb(root)
    if found is None:
        raise FileNotFoundError(f"No se encontró la carpeta aclImdb dentro de {root}")
    return found


def load_dataframe(data_dir):
    """Lee las 50.000 reseñas (train + test originales) en un DataFrame."""
    data_dir = Path(data_dir)
    rows = []
    for split in ("train", "test"):
        for label, name in enumerate(("neg", "pos")):
            for f in sorted((data_dir / split / name).glob("*.txt")):
                rows.append((f.read_text(encoding="utf-8"), label, split))
    df = pd.DataFrame(rows, columns=["text", "label", "original_split"])
    df["sentimiento"] = df["label"].map(dict(enumerate(CLASS_NAMES)))
    return df


def clean_text(text):
    """Quita etiquetas HTML (<br />), URLs y espacios repetidos.
    Minúsculas y puntuación las maneja la capa TextVectorization dentro del modelo."""
    text = _TAGS.sub(" ", text)
    text = _URLS.sub(" ", text)
    text = text.replace("&amp;", "&")
    return _SPACES.sub(" ", text).strip()


def split_data(df, val_size=0.15, test_size=0.15, seed=SEED):
    """División estratificada 70/15/15. Devuelve dict de DataFrames."""
    train, tmp = train_test_split(df, test_size=val_size + test_size,
                                  stratify=df["label"], random_state=seed)
    val, test = train_test_split(tmp, test_size=test_size / (val_size + test_size),
                                 stratify=tmp["label"], random_state=seed)
    return {"train": train.reset_index(drop=True),
            "val": val.reset_index(drop=True),
            "test": test.reset_index(drop=True)}


def make_dataset(texts, labels, batch_size=64, training=False):
    ds = tf.data.Dataset.from_tensor_slices((list(texts), list(map(float, labels))))
    if training:
        ds = ds.shuffle(len(texts), seed=SEED)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
