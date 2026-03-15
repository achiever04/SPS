"""
===============================================================================
  inference_engine.py — ONNX Runtime Model Inference Layer
===============================================================================
  Loads a pre-trained ONNX model and StandardScaler at startup, then provides
  sub-millisecond predictions on streaming feature vectors.

  ⚡ NO PyTorch / NO TensorFlow — onnxruntime only.
  ⚡ Single InferenceSession, reused across all predictions.
  ⚡ Latency logged on every prediction for performance monitoring.

  Architecture:
      feature_engine → numpy vector → InferenceEngine.predict() → signal
===============================================================================
"""

import time
import os
from typing import Optional

import numpy as np

from utils.logger import get_logger, log_latency

# ── Module logger ───────────────────────────────────────────────────────
logger = get_logger("INFERENCE")


class InferenceEngine:
    """
    Production ONNX inference engine for real-time trading signal generation.

    The engine:
    1. Loads model.onnx + scaler.pkl once at startup.
    2. Scales incoming feature vectors using the pre-fitted scaler.
    3. Runs ONNX inference session for sub-ms predictions.
    4. Returns a trading signal: BUY (+1), SELL (-1), or HOLD (0).

    Attributes:
        model_path:    Path to the .onnx model file.
        scaler_path:   Path to the .pkl scaler file.
        _session:      onnxruntime.InferenceSession instance.
        _scaler:       Pre-fitted StandardScaler (or equivalent).
        _input_name:   ONNX model input tensor name.
        _output_name:  ONNX model output tensor name.
        _total_inferences: Counter for total predictions made.
        _total_latency_ns: Cumulative latency for average tracking.
    """

    # ── Signal constants ─────────────────────────────────────────────────
    SIGNAL_BUY: int = 1
    SIGNAL_SELL: int = -1
    SIGNAL_HOLD: int = 0

    def __init__(self, model_path: str, scaler_path: str):
        """
        Initialize the inference engine by loading model and scaler.

        Args:
            model_path:  Absolute path to the ONNX model file.
            scaler_path: Absolute path to the scaler pickle file.

        Raises:
            FileNotFoundError: If model or scaler files don't exist.
            RuntimeError: If ONNX session creation fails.
        """
        self.model_path = model_path
        self.scaler_path = scaler_path

        # ── Runtime state ────────────────────────────────────────────────
        self._session = None
        self._scaler = None
        self._input_name: str = ""
        self._output_name: str = ""
        self._total_inferences: int = 0
        self._total_latency_ns: int = 0

        # ── Confidence thresholds ────────────────────────────────────────
        self._buy_threshold: float = 0.6    # Probability > 0.6 → BUY
        self._sell_threshold: float = 0.4   # Probability < 0.4 → SELL

        # ── Load model and scaler ────────────────────────────────────────
        self._load_model()
        self._load_scaler()

    def _load_model(self) -> None:
        """
        Load the ONNX model into an InferenceSession.

        Uses CPU execution provider with optimizations enabled.
        Graph optimization is set to ORT_ENABLE_ALL for maximum
        inference speed.
        """
        if not os.path.exists(self.model_path):
            logger.warning(
                f"ONNX model not found at {self.model_path}. "
                f"Inference will run in DUMMY mode (always returns HOLD)."
            )
            return

        try:
            import onnxruntime as ort

            t_start = time.perf_counter_ns()

            # ── Configure session options for minimal latency ────────────
            session_options = ort.SessionOptions()
            session_options.graph_optimization_level = (
                ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            )
            # ── Use a single thread for deterministic latency ────────────
            session_options.intra_op_num_threads = 1
            session_options.inter_op_num_threads = 1
            # ── Disable memory pattern for lower memory usage ────────────
            session_options.enable_mem_pattern = False

            self._session = ort.InferenceSession(
                self.model_path,
                sess_options=session_options,
                providers=["CPUExecutionProvider"],
            )

            # ── Cache input/output tensor names ──────────────────────────
            self._input_name = self._session.get_inputs()[0].name
            self._output_name = self._session.get_outputs()[0].name

            elapsed_ms = log_latency(logger, "MODEL_LOAD", t_start)
            logger.info(
                f"✅ ONNX model loaded | "
                f"Input: {self._input_name} | "
                f"Output: {self._output_name} | "
                f"Load time: {elapsed_ms:.1f}ms"
            )

        except ImportError:
            logger.error(
                "onnxruntime not installed. "
                "Install with: pip install onnxruntime"
            )
        except Exception as e:
            logger.exception(f"Failed to load ONNX model: {e}")

    def _load_scaler(self) -> None:
        """
        Load the pre-fitted scaler (StandardScaler / MinMaxScaler)
        from a pickle or joblib file.

        The scaler must have been fitted on the same features
        used during model training.
        """
        if not os.path.exists(self.scaler_path):
            logger.warning(
                f"Scaler not found at {self.scaler_path}. "
                f"Features will NOT be scaled (raw values used)."
            )
            return

        try:
            import joblib

            t_start = time.perf_counter_ns()
            self._scaler = joblib.load(self.scaler_path)
            log_latency(logger, "SCALER_LOAD", t_start)
            logger.info("✅ Feature scaler loaded successfully")

        except ImportError:
            logger.error(
                "joblib not installed. Install with: pip install joblib"
            )
        except Exception as e:
            logger.exception(f"Failed to load scaler: {e}")

    def predict(self, features: np.ndarray) -> dict:
        """
        Run inference on a feature vector and return a trading signal.

        Args:
            features: 1D numpy array of shape (n_features,) from FeatureEngine.

        Returns:
            A dict containing:
            {
                "signal": int,          # +1 (BUY), -1 (SELL), 0 (HOLD)
                "confidence": float,    # Model confidence (0.0 - 1.0)
                "latency_ms": float,    # Inference latency in milliseconds
                "raw_output": any,      # Raw model output for debugging
            }

        Performance:
            Target: < 1ms per prediction on CPU.
        """
        t_start = time.perf_counter_ns()

        # ── Fallback: No model loaded → HOLD ────────────────────────────
        if self._session is None:
            return {
                "signal": self.SIGNAL_HOLD,
                "confidence": 0.0,
                "latency_ms": 0.0,
                "raw_output": None,
            }

        try:
            # ── Step 1: Scale features ───────────────────────────────────
            # Reshape to (1, n_features) for scaler and ONNX
            input_data = features.reshape(1, -1)

            if self._scaler is not None:
                input_data = self._scaler.transform(input_data)

            # ── Ensure correct dtype (ONNX expects float32) ──────────────
            input_data = input_data.astype(np.float32)

            # ── Step 2: Run ONNX inference ───────────────────────────────
            raw_output = self._session.run(
                [self._output_name],
                {self._input_name: input_data},
            )

            # ── Step 3: Interpret output ─────────────────────────────────
            signal, confidence = self._interpret_output(raw_output)

            # ── Step 4: Track performance ────────────────────────────────
            elapsed_ns = time.perf_counter_ns() - t_start
            elapsed_ms = elapsed_ns / 1_000_000
            self._total_inferences += 1
            self._total_latency_ns += elapsed_ns

            # Log every 100th inference to avoid log flooding
            if self._total_inferences % 100 == 0:
                avg_ms = (self._total_latency_ns / self._total_inferences) / 1_000_000
                logger.info(
                    f"📊 Inference #{self._total_inferences} | "
                    f"This: {elapsed_ms:.3f}ms | "
                    f"Avg: {avg_ms:.3f}ms | "
                    f"Signal: {signal} ({confidence:.2%})"
                )

            return {
                "signal": signal,
                "confidence": confidence,
                "latency_ms": elapsed_ms,
                "raw_output": raw_output,
            }

        except Exception as e:
            logger.exception(f"Inference error: {e}")
            return {
                "signal": self.SIGNAL_HOLD,
                "confidence": 0.0,
                "latency_ms": (time.perf_counter_ns() - t_start) / 1_000_000,
                "raw_output": None,
            }

    def _interpret_output(
        self, raw_output: list
    ) -> tuple[int, float]:
        """
        Convert raw ONNX model output to a trading signal.

        Supports two output formats:
        1. Classification: Output is class probabilities [P(sell), P(hold), P(buy)]
        2. Regression: Output is a single value (positive → buy, negative → sell)

        Args:
            raw_output: List of numpy arrays from ONNX session.run().

        Returns:
            Tuple of (signal: int, confidence: float).
        """
        output = raw_output[0]

        # ── Classification output (e.g., softmax probabilities) ──────────
        if output.ndim == 2 and output.shape[1] >= 2:
            probabilities = output[0]

            if len(probabilities) == 3:
                # [P(sell), P(hold), P(buy)]
                predicted_class = int(np.argmax(probabilities))
                confidence = float(probabilities[predicted_class])
                signal_map = {0: self.SIGNAL_SELL, 1: self.SIGNAL_HOLD, 2: self.SIGNAL_BUY}
                return signal_map.get(predicted_class, self.SIGNAL_HOLD), confidence

            elif len(probabilities) == 2:
                # [P(sell/hold), P(buy)]
                buy_prob = float(probabilities[1])
                if buy_prob > self._buy_threshold:
                    return self.SIGNAL_BUY, buy_prob
                elif buy_prob < self._sell_threshold:
                    return self.SIGNAL_SELL, 1.0 - buy_prob
                else:
                    return self.SIGNAL_HOLD, 1.0 - abs(buy_prob - 0.5) * 2

        # ── Regression output (single value) ─────────────────────────────
        elif output.ndim <= 2:
            value = float(output.flat[0])
            confidence = min(abs(value), 1.0)

            if value > self._buy_threshold:
                return self.SIGNAL_BUY, confidence
            elif value < -self._buy_threshold:
                return self.SIGNAL_SELL, confidence
            else:
                return self.SIGNAL_HOLD, 1.0 - confidence

        # ── Unknown format → HOLD ────────────────────────────────────────
        logger.warning(f"Unknown model output shape: {output.shape}")
        return self.SIGNAL_HOLD, 0.0

    # ── Diagnostics ──────────────────────────────────────────────────────

    @property
    def is_ready(self) -> bool:
        """Return True if model is loaded and ready for inference."""
        return self._session is not None

    @property
    def stats(self) -> dict:
        """Return inference performance statistics."""
        avg_ms = 0.0
        if self._total_inferences > 0:
            avg_ms = (self._total_latency_ns / self._total_inferences) / 1_000_000

        return {
            "total_inferences": self._total_inferences,
            "avg_latency_ms": round(avg_ms, 4),
            "model_loaded": self._session is not None,
            "scaler_loaded": self._scaler is not None,
        }
