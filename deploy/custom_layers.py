"""Custom TensorFlow/Keras layers and Seq2Seq model matching model_forecasting.ipynb.

These definitions mirror Cell 20 & Cell 24 of `model_forecasting.ipynb` so the
trained `.keras` artifacts can be loaded cleanly without modifying the original
notebook or model files.
"""

from __future__ import annotations

import tensorflow as tf
from tensorflow.keras.layers import Layer, LSTM
from tensorflow.keras.models import Model


class CustomDense(Layer):
    def __init__(self, units: int, **kwargs):
        super().__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        self.w = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer="glorot_uniform",
            trainable=True,
            name="kernel",
        )
        self.b = self.add_weight(
            shape=(self.units,),
            initializer="zeros",
            trainable=True,
            name="bias",
        )
        super().build(input_shape)

    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b

    def get_config(self):
        config = super().get_config()
        config.update({"units": self.units})
        return config


class CustomMultiHeadAttention(Layer):
    def __init__(self, num_heads: int, head_dim: int, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.d_model = num_heads * head_dim

    def build(self, input_shape):
        self.wq = tf.keras.layers.Dense(self.d_model)
        self.wk = tf.keras.layers.Dense(self.d_model)
        self.wv = tf.keras.layers.Dense(self.d_model)
        self.dense = tf.keras.layers.Dense(input_shape[-1])
        super().build(input_shape)

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.head_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, q, k, v):
        batch_size = tf.shape(q)[0]
        seq_len = q.shape[1]

        q_proj, k_proj, v_proj = self.wq(q), self.wk(k), self.wv(v)
        q_proj = self.split_heads(q_proj, batch_size)
        k_proj = self.split_heads(k_proj, batch_size)
        v_proj = self.split_heads(v_proj, batch_size)

        matmul_qk = tf.matmul(q_proj, k_proj, transpose_b=True)
        dk = tf.cast(tf.shape(k_proj)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
        output = tf.matmul(attention_weights, v_proj)

        output = tf.transpose(output, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(output, (batch_size, -1, self.d_model))
        final_output = self.dense(concat_attention)

        final_output.set_shape((None, seq_len, final_output.shape[-1]))
        return final_output

    def get_config(self):
        config = super().get_config()
        config.update({"num_heads": self.num_heads, "head_dim": self.head_dim})
        return config


class CustomLeakyReLU(Layer):
    def __init__(self, alpha: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha

    def call(self, inputs):
        return tf.maximum(self.alpha * inputs, inputs)

    def get_config(self):
        config = super().get_config()
        config.update({"alpha": self.alpha})
        return config


class Seq2SeqTeacherForcing(Model):
    """Encoder-Decoder LSTM with Cross-Attention.

    Default `units=128, horizon=24` matches the saved `model_seq2seq_LSTM.keras`
    artifact whose config.json omits those constructor arguments.
    """

    def __init__(self, units: int = 128, horizon: int = 24, **kwargs):
        super().__init__(**kwargs)
        self.horizon = horizon
        self.units = units

        self.encoder_lstm = LSTM(units, return_sequences=True, return_state=True)
        self.attention = CustomMultiHeadAttention(num_heads=2, head_dim=32)
        self.decoder_lstm = LSTM(units, return_sequences=True, return_state=True)
        self.custom_dense = CustomDense(units=1)

    def call(self, inputs, targets=None, training=False):
        enc_outputs, state_h, state_c = self.encoder_lstm(inputs)
        dec_input = tf.reshape(inputs[:, -1, 0], [-1, 1, 1])

        predictions = []
        for t in range(self.horizon):
            dec_outputs, state_h, state_c = self.decoder_lstm(
                dec_input, initial_state=[state_h, state_c]
            )
            context_vector = self.attention(dec_outputs, enc_outputs, enc_outputs)
            combined = tf.concat([dec_outputs, context_vector], axis=-1)
            pred = self.custom_dense(combined)

            pred_flat = tf.reshape(pred, [-1, 1])
            predictions.append(pred_flat)

            if training and targets is not None:
                dec_input = tf.reshape(targets[:, t], [-1, 1, 1])
            else:
                dec_input = tf.reshape(pred, [-1, 1, 1])

        predictions = tf.stack(predictions, axis=1)
        predictions = tf.squeeze(predictions, axis=-1)
        return predictions

    def get_config(self):
        config = super().get_config()
        config.update({"units": self.units, "horizon": self.horizon})
        return config


CUSTOM_OBJECTS = {
    "CustomDense": CustomDense,
    "CustomMultiHeadAttention": CustomMultiHeadAttention,
    "CustomLeakyReLU": CustomLeakyReLU,
    "Seq2SeqTeacherForcing": Seq2SeqTeacherForcing,
}
