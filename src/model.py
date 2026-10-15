"""Modelo secuencial: TextVectorization -> Embedding -> RNN bidireccional (LSTM/GRU) -> Dense."""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

MAX_TOKENS = 20000   # tamaño del vocabulario
SEQ_LEN = 300        # longitud fija de cada secuencia (cubre ~75 % de las reseñas completas)


def build_vectorizer(train_texts, max_tokens=MAX_TOKENS, seq_len=SEQ_LEN):
    """Crea y ajusta (adapt) el vocabulario SOLO con el conjunto de entrenamiento."""
    vec = layers.TextVectorization(max_tokens=max_tokens, output_mode="int",
                                   output_sequence_length=seq_len, name="vectorizer")
    vec.adapt(tf.data.Dataset.from_tensor_slices(list(train_texts)).batch(256))
    return vec


def build_model(vectorizer, cell="lstm", embed_dim=64, units=64):
    """El vectorizador va DENTRO del modelo: la entrada es texto crudo (string)."""
    rnn = {"lstm": layers.LSTM, "gru": layers.GRU, "simple": layers.SimpleRNN}[cell]
    inputs = keras.Input(shape=(), dtype="string", name="texto")
    x = vectorizer(inputs)
    x = layers.Embedding(vectorizer.vocabulary_size(), embed_dim, mask_zero=True)(x)
    x = layers.Bidirectional(rnn(units))(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)  # P(reseña positiva)

    model = keras.Model(inputs, outputs, name=f"sentimiento_{cell}")
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="binary_crossentropy",
                  metrics=["accuracy", keras.metrics.AUC(name="auc")])
    return model
