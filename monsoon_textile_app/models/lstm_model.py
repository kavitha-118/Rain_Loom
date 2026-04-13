"""
LSTM Regime Predictor for Monsoon-Textile Volatility System.

Sequence-based deep learning models for predicting high-volatility regimes
in Indian textile equities.  Includes a standard stacked-LSTM architecture
and an attention-augmented variant that exposes temporal attention weights
for interpretability.

All train/validation/test splits are strictly chronological.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import backend as K
    from tensorflow.keras.callbacks import (
        EarlyStopping,
        ModelCheckpoint,
        ReduceLROnPlateau,
    )
    from tensorflow.keras.layers import (
        BatchNormalization,
        Dense,
        Dropout,
        Input,
        LSTM,
        Layer,
        Multiply,
        Permute,
        RepeatVector,
    )
    from tensorflow.keras.models import Model, Sequential, load_model
    _HAS_TF = True
except ImportError:
    _HAS_TF = False
    logger.warning("TensorFlow not installed — LSTM layer disabled")


# ---------------------------------------------------------------------------
# Default configuration
# ---------------------------------------------------------------------------
_DEFAULT_CONFIG: Dict[str, Any] = {
    "seq_length": 12,
    "lstm_units_1": 64,
    "lstm_units_2": 32,
    "dense_units": 16,
    "dropout_rate": 0.3,
    "learning_rate": 1e-3,
    "batch_size": 32,
    "epochs": 200,
    "patience_early_stop": 15,
    "patience_reduce_lr": 7,
    "min_lr": 1e-6,
    "random_seed": 42,
}


# ===========================================================================
# Custom Temporal Attention Layer
# ===========================================================================
class TemporalAttentionLayer(Layer):
    """Learns a scalar attention weight per timestep in a sequence.

    Given an input of shape ``(batch, seq_length, features)``, this layer
    produces a weighted sum over the time axis, highlighting the weeks that
    matter most for regime prediction.

    Attributes
    ----------
    attention_dense : Dense
        Learnable projection to a single score per timestep.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.attention_dense: Optional[Dense] = None

    def build(self, input_shape: tf.TensorShape) -> None:
        """Create the trainable dense layer on first call.

        Parameters
        ----------
        input_shape : tf.TensorShape
            Expected ``(batch, seq_length, features)``.
        """
        self.attention_dense = Dense(1, activation="tanh", name="attn_score")
        super().build(input_shape)

    def call(self, inputs: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """Forward pass.

        Parameters
        ----------
        inputs : tf.Tensor
            Shape ``(batch, seq_length, features)``.

        Returns
        -------
        context : tf.Tensor
            Weighted sum over time, shape ``(batch, features)``.
        weights : tf.Tensor
            Normalised attention weights, shape ``(batch, seq_length, 1)``.
        """
        # (batch, seq_length, 1)
        scores = self.attention_dense(inputs)
        weights = tf.nn.softmax(scores, axis=1)
        # (batch, features)
        context = tf.reduce_sum(inputs * weights, axis=1)
        return context, weights

    def get_config(self) -> Dict[str, Any]:
        """Serialisation config for Keras model saving."""
        return super().get_config()


# ===========================================================================
# LSTM Regime Predictor
# ===========================================================================
class LSTMRegimePredictor:
    """LSTM-based binary classifier for high-volatility textile regimes.

    Parameters
    ----------
    config : dict, optional
        Architecture and training hyper-parameters.  Missing keys fall back
        to ``_DEFAULT_CONFIG``.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config: Dict[str, Any] = {**_DEFAULT_CONFIG, **(config or {})}
        self.model: Optional[keras.Model] = None
        self.attention_model: Optional[Model] = None
        self.scaler: Optional[StandardScaler] = None
        self.history: Optional[keras.callbacks.History] = None
        self._is_attention: bool = False

        tf.random.set_seed(self.config["random_seed"])
        logger.info(
            "LSTMRegimePredictor initialised | seq_length={} lstm1={} lstm2={}",
            self.config["seq_length"],
            self.config["lstm_units_1"],
            self.config["lstm_units_2"],
        )

    # ------------------------------------------------------------------
    # Sequence creation
    # ------------------------------------------------------------------
    def create_sequences(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        target_col: str = "high_vol_regime",
        seq_length: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
        """Convert a 2-D DataFrame into 3-D sequence arrays for LSTM input.

        Parameters
        ----------
        df : pd.DataFrame
            Source data, assumed sorted chronologically.
        feature_cols : list[str]
            Columns to use as input features.
        target_col : str
            Binary regime column.
        seq_length : int, optional
            Lookback window (defaults to ``config['seq_length']``).

        Returns
        -------
        X_seq : np.ndarray
            Shape ``(n_samples, seq_length, n_features)``.
        y_seq : np.ndarray
            Shape ``(n_samples,)``.
        scaler : StandardScaler
            Fitted scaler (must be reused at inference time).
        """
        seq_length = seq_length or self.config["seq_length"]

        # Scale features -- fit only on the data provided (caller is
        # responsible for passing *training* data here to avoid leakage).
        self.scaler = StandardScaler()
        features = self.scaler.fit_transform(df[feature_cols].values)
        targets = df[target_col].values

        X_seq: List[np.ndarray] = []
        y_seq: List[int] = []

        for i in range(seq_length, len(features)):
            X_seq.append(features[i - seq_length : i])
            y_seq.append(int(targets[i]))

        X_arr = np.array(X_seq, dtype=np.float32)
        y_arr = np.array(y_seq, dtype=np.float32)

        logger.info(
            "create_sequences | X_seq={} y_seq={} pos_rate={:.3f}",
            X_arr.shape,
            y_arr.shape,
            y_arr.mean(),
        )
        return X_arr, y_arr, self.scaler

    # ------------------------------------------------------------------
    # Standard stacked-LSTM
    # ------------------------------------------------------------------
    def build_model(
        self,
        n_features: int,
        seq_length: Optional[int] = None,
    ) -> Sequential:
        """Build and compile a stacked-LSTM binary classifier.

        Architecture
        ------------
        LSTM(64) -> Dropout -> BatchNorm -> LSTM(32) -> Dropout
        -> Dense(16, relu) -> Dense(1, sigmoid)

        Parameters
        ----------
        n_features : int
            Number of input features per timestep.
        seq_length : int, optional
            Lookback window.

        Returns
        -------
        keras.Sequential
            Compiled model.
        """
        seq_length = seq_length or self.config["seq_length"]
        cfg = self.config

        model = Sequential(
            [
                LSTM(
                    cfg["lstm_units_1"],
                    return_sequences=True,
                    input_shape=(seq_length, n_features),
                    name="lstm_1",
                ),
                Dropout(cfg["dropout_rate"], name="dropout_1"),
                BatchNormalization(name="bn_1"),
                LSTM(cfg["lstm_units_2"], return_sequences=False, name="lstm_2"),
                Dropout(cfg["dropout_rate"], name="dropout_2"),
                Dense(cfg["dense_units"], activation="relu", name="dense_hidden"),
                Dense(1, activation="sigmoid", name="output"),
            ],
            name="lstm_regime_classifier",
        )

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=cfg["learning_rate"]),
            loss="binary_crossentropy",
            metrics=[
                "accuracy",
                keras.metrics.AUC(name="auc"),
                keras.metrics.Precision(name="precision"),
                keras.metrics.Recall(name="recall"),
            ],
        )

        self.model = model
        self._is_attention = False
        logger.info("Standard LSTM model built | params={:,}", model.count_params())
        return model

    # ------------------------------------------------------------------
    # Attention-augmented LSTM
    # ------------------------------------------------------------------
    def build_attention_model(
        self,
        n_features: int,
        seq_length: Optional[int] = None,
    ) -> Model:
        """Build an LSTM model with a temporal attention mechanism.

        The attention layer learns which of the ``seq_length`` weeks
        contribute most to the final prediction.

        Parameters
        ----------
        n_features : int
            Number of input features per timestep.
        seq_length : int, optional
            Lookback window.

        Returns
        -------
        keras.Model
            Compiled functional-API model with two outputs:
            ``prediction`` and ``attention_weights``.
        """
        seq_length = seq_length or self.config["seq_length"]
        cfg = self.config

        inp = Input(shape=(seq_length, n_features), name="sequence_input")

        x = LSTM(
            cfg["lstm_units_1"], return_sequences=True, name="lstm_1"
        )(inp)
        x = Dropout(cfg["dropout_rate"], name="dropout_1")(x)
        x = BatchNormalization(name="bn_1")(x)

        x = LSTM(
            cfg["lstm_units_2"], return_sequences=True, name="lstm_2"
        )(x)
        x = Dropout(cfg["dropout_rate"], name="dropout_2")(x)

        # Temporal attention
        attn_layer = TemporalAttentionLayer(name="temporal_attention")
        context, attn_weights = attn_layer(x)

        x = Dense(cfg["dense_units"], activation="relu", name="dense_hidden")(context)
        prediction = Dense(1, activation="sigmoid", name="prediction")(x)

        # Attention weights output for interpretability
        attn_out = keras.layers.Lambda(
            lambda t: t, name="attention_weights"
        )(attn_weights)

        model = Model(
            inputs=inp,
            outputs=[prediction, attn_out],
            name="lstm_attention_regime",
        )

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=cfg["learning_rate"]),
            loss={
                "prediction": "binary_crossentropy",
                "attention_weights": None,  # no loss on attention output
            },
            metrics={
                "prediction": [
                    "accuracy",
                    keras.metrics.AUC(name="auc"),
                ],
            },
        )

        self.model = model
        self.attention_model = model
        self._is_attention = True
        logger.info("Attention LSTM model built | params={:,}", model.count_params())
        return model

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        checkpoint_path: Optional[str] = None,
    ) -> Tuple[keras.Model, keras.callbacks.History]:
        """Train the model with class-weight balancing and callbacks.

        Parameters
        ----------
        X_train, y_train : np.ndarray
            Training sequences and labels.
        X_val, y_val : np.ndarray
            Chronological validation sequences and labels.
        checkpoint_path : str, optional
            Path to save the best model checkpoint.

        Returns
        -------
        model : keras.Model
            Trained model.
        history : keras.callbacks.History
            Training history for plotting.
        """
        if self.model is None:
            raise RuntimeError("Model not built. Call build_model() first.")

        cfg = self.config

        # Class weights for imbalanced regime labels
        classes = np.unique(y_train)
        weights = compute_class_weight("balanced", classes=classes, y=y_train)
        class_weight_dict = {int(c): float(w) for c, w in zip(classes, weights)}
        logger.info("Class weights: {}", class_weight_dict)

        callbacks = [
            EarlyStopping(
                monitor="val_loss",
                patience=cfg["patience_early_stop"],
                restore_best_weights=True,
                verbose=1,
            ),
            ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=cfg["patience_reduce_lr"],
                min_lr=cfg["min_lr"],
                verbose=1,
            ),
        ]

        if checkpoint_path:
            callbacks.append(
                ModelCheckpoint(
                    checkpoint_path,
                    monitor="val_loss",
                    save_best_only=True,
                    verbose=0,
                )
            )

        # Prepare targets depending on model type
        if self._is_attention:
            # Attention model has two outputs; attention_weights has no loss,
            # but Keras still expects a target array.
            seq_len = X_train.shape[1]
            dummy_attn_train = np.zeros(
                (len(y_train), seq_len, 1), dtype=np.float32
            )
            dummy_attn_val = np.zeros(
                (len(y_val), seq_len, 1), dtype=np.float32
            )
            y_fit = [y_train, dummy_attn_train]
            y_val_fit = [y_val, dummy_attn_val]
        else:
            y_fit = y_train
            y_val_fit = y_val

        self.history = self.model.fit(
            X_train,
            y_fit,
            validation_data=(X_val, y_val_fit),
            epochs=cfg["epochs"],
            batch_size=cfg["batch_size"],
            class_weight=class_weight_dict if not self._is_attention else None,
            sample_weight=(
                self._compute_sample_weights(y_train)
                if self._is_attention
                else None
            ),
            callbacks=callbacks,
            verbose=1,
        )

        best_epoch = (
            np.argmin(self.history.history["val_loss"]) + 1
            if "val_loss" in self.history.history
            else cfg["epochs"]
        )
        logger.info(
            "Training complete | best_epoch={} final_val_loss={:.4f}",
            best_epoch,
            min(self.history.history.get("val_loss", [float("inf")])),
        )
        return self.model, self.history

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return regime probabilities for input sequences.

        Parameters
        ----------
        X : np.ndarray
            Shape ``(n_samples, seq_length, n_features)``.

        Returns
        -------
        np.ndarray
            1-D array of probabilities in [0, 1].
        """
        if self.model is None:
            raise RuntimeError("No trained model available.")

        raw = self.model.predict(X, verbose=0)

        if self._is_attention:
            # First output is the prediction, second is attention weights
            proba = np.asarray(raw[0]).flatten()
        else:
            proba = np.asarray(raw).flatten()

        return proba

    # ------------------------------------------------------------------
    # Attention interpretability
    # ------------------------------------------------------------------
    def attention_weights(self, X: np.ndarray) -> np.ndarray:
        """Extract temporal attention weights for each input sequence.

        Parameters
        ----------
        X : np.ndarray
            Shape ``(n_samples, seq_length, n_features)``.

        Returns
        -------
        np.ndarray
            Shape ``(n_samples, seq_length)`` -- normalised weights
            summing to 1 across the time axis.

        Raises
        ------
        RuntimeError
            If the model was not built with ``build_attention_model()``.
        """
        if not self._is_attention or self.attention_model is None:
            raise RuntimeError(
                "Attention weights are only available for the attention model. "
                "Call build_attention_model() and train first."
            )

        raw = self.attention_model.predict(X, verbose=0)
        # raw[1] has shape (n_samples, seq_length, 1)
        weights = np.asarray(raw[1]).squeeze(-1)
        return weights

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, path: str | Path) -> None:
        """Save the full model, scaler, and config to a directory.

        Parameters
        ----------
        path : str or Path
            Directory where artefacts will be written.
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        if self.model is not None:
            model_path = path / "model.keras"
            self.model.save(
                model_path,
                custom_objects={"TemporalAttentionLayer": TemporalAttentionLayer},
            )
            logger.info("Keras model saved to {}", model_path)

        if self.scaler is not None:
            import joblib

            joblib.dump(self.scaler, path / "scaler.joblib")

        # Save config + metadata
        meta = {
            "config": self.config,
            "is_attention": self._is_attention,
        }
        with open(path / "meta.json", "w") as fh:
            json.dump(meta, fh, indent=2)

        logger.info("All artefacts saved to {}", path)

    @classmethod
    def load(cls, path: str | Path) -> "LSTMRegimePredictor":
        """Load a previously saved predictor.

        Parameters
        ----------
        path : str or Path
            Directory containing saved artefacts.

        Returns
        -------
        LSTMRegimePredictor
            Restored instance.
        """
        path = Path(path)

        with open(path / "meta.json") as fh:
            meta = json.load(fh)

        instance = cls(config=meta["config"])
        instance._is_attention = meta["is_attention"]

        model_path = path / "model.keras"
        if model_path.exists():
            instance.model = load_model(
                model_path,
                custom_objects={"TemporalAttentionLayer": TemporalAttentionLayer},
            )
            if instance._is_attention:
                instance.attention_model = instance.model
            logger.info("Keras model loaded from {}", model_path)

        scaler_path = path / "scaler.joblib"
        if scaler_path.exists():
            import joblib

            instance.scaler = joblib.load(scaler_path)

        logger.info("LSTMRegimePredictor restored from {}", path)
        return instance

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _compute_sample_weights(y: np.ndarray) -> np.ndarray:
        """Compute per-sample weights for imbalanced classes.

        Used when ``class_weight`` cannot be applied directly (e.g.
        multi-output attention model).
        """
        classes = np.unique(y)
        weights = compute_class_weight("balanced", classes=classes, y=y)
        weight_map = {int(c): float(w) for c, w in zip(classes, weights)}
        return np.array([weight_map[int(label)] for label in y], dtype=np.float32)
