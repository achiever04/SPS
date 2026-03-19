"""
===============================================================================
  inference_engine.py — ONNX Runtime Model Inference Layer  (v2)
===============================================================================
  Loads a pre-trained ONNX model and fitted scaler at startup, then provides
  sub-millisecond predictions on streaming feature vectors.

  Changes in v2:
  • Hard failure when model file is missing (no silent HOLD fallback)
    unless dry_run=True is explicitly passed.
  • Unified confidence threshold logic — buy_threshold / sell_threshold
    used consistently in both classification and regression branches.
  • MIN_CONFIDENCE gate — signals with low confidence are suppressed here
    rather than relying on the execution engine alone.
  • Inference stats track p50 / p95 latency (ring buffer).
  • Thread-safe prediction counter with lock.

  Architecture:
      feature_engine → numpy (19,) → InferenceEngine.predict() → signal dict
===============================================================================
"""

import os
import threading
import time
from collections import deque
from typing import Optional

import numpy as np

from utils.logger import get_logger, log_latency

logger = get_logger("INFERENCE")


class InferenceEngine:
    """
    Production ONNX inference engine for real-time trading signals.

    Steps per prediction:
    1. Validate input shape matches the loaded model.
    2. Scale features via the pre-fitted scaler.
    3. Run ONNX InferenceSession (CPU, single-threaded for determinism).
    4. Decode class probabilities / regression value → signal + confidence.
    5. Apply MIN_CONFIDENCE gate.

    Attributes:
        model_path:        Path to .onnx model file.
        scaler_path:       Path to scaler .pkl file.
        min_confidence:    Reject signals below this probability.
        dry_run:           If True, allow missing model (always returns HOLD).
    """

    SIGNAL_BUY:  int = 1
    SIGNAL_SELL: int = -1
    SIGNAL_HOLD: int = 0

    def __init__(
        self,
        model_path: str,
        scaler_path: str,
        min_confidence: float = 0.60,
        buy_threshold: float = 0.60,
        sell_threshold: float = 0.40,
        dry_run: bool = False,
    ):
        """
        Args:
            model_path:      Absolute path to the ONNX model.
            scaler_path:     Absolute path to the scaler pickle.
            min_confidence:  Minimum probability to emit a non-HOLD signal.
            buy_threshold:   Probability above which BUY is triggered.
            sell_threshold:  Probability below which SELL is triggered.
            dry_run:         When True, missing model is a warning not a crash.
        """
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.min_confidence = min_confidence
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        self.dry_run = dry_run

        # ── Runtime objects ──────────────────────────────────────────────
        self._session = None
        self._scaler = None
        self._input_name: str = ""
        self._output_name: str = ""
        self._expected_features: Optional[int] = None

        # ── Performance counters ─────────────────────────────────────────
        self._lock = threading.Lock()
        self._total_inferences: int = 0
        self._total_latency_ns: int = 0
        # Ring buffer of last 200 latencies (ns) for p50/p95 tracking
        self._latency_ring: deque = deque(maxlen=200)

        # Signal distribution counters
        self._signal_counts = {self.SIGNAL_BUY: 0, self.SIGNAL_SELL: 0, self.SIGNAL_HOLD: 0}

        # ── Load model and scaler ────────────────────────────────────────
        self._load_model()
        self._load_scaler()

    # ═════════════════════════════════════════════════════════════════════
    #  MODEL / SCALER LOADING
    # ═════════════════════════════════════════════════════════════════════

    def _load_model(self) -> None:
        """
        Load ONNX model into an InferenceSession with full graph
        optimization and single-threaded CPU execution.

        Raises SystemExit when the model is missing and dry_run=False.
        """
        if not os.path.exists(self.model_path):
            msg = (
                f"ONNX model not found at {self.model_path}. "
                f"Run scripts/train_model.py first."
            )
            if self.dry_run:
                logger.warning(f"{msg} Running in DRY-RUN / HOLD-only mode.")
                return
            else:
                logger.critical(msg)
                raise FileNotFoundError(msg)

        try:
            import onnxruntime as ort

            t_start = time.perf_counter_ns()

            opts = ort.SessionOptions()
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            opts.intra_op_num_threads = 1
            opts.inter_op_num_threads = 1
            opts.enable_mem_pattern = False
            opts.enable_cpu_mem_arena = False   # Avoid memory fragmentation

            self._session = ort.InferenceSession(
                self.model_path,
                sess_options=opts,
                providers=["CPUExecutionProvider"],
            )

            self._input_name = self._session.get_inputs()[0].name
            all_outputs = [o.name for o in self._session.get_outputs()]
            # Prefer probabilities output over label output
            self._output_name = next(
                (o for o in all_outputs if "prob" in o.lower()),
                all_outputs[-1]
            )
            logger.info(f"   Model outputs available: {all_outputs}")
            logger.info(f"   Using output: '{self._output_name}'")
            # Cache expected input feature count for runtime validation
            self._expected_features = self._session.get_inputs()[0].shape[1]

            elapsed_ms = log_latency(logger, "MODEL_LOAD", t_start)
            logger.info(
                f"✅ ONNX model loaded | "
                f"Input: '{self._input_name}' ({self._expected_features} features) | "
                f"Output: '{self._output_name}' | "
                f"Load: {elapsed_ms:.1f}ms"
            )

        except ImportError:
            raise ImportError(
                "onnxruntime not installed. Run: pip install onnxruntime"
            )
        except Exception as e:
            logger.exception(f"Failed to load ONNX model: {e}")
            raise

    def _load_scaler(self) -> None:
        """
        Load the pre-fitted scaler (StandardScaler / MinMaxScaler).

        A missing scaler is a warning, not a crash — raw features will
        be used, which degrades accuracy but keeps the bot alive.
        """
        if not os.path.exists(self.scaler_path):
            logger.warning(
                f"Scaler not found at {self.scaler_path}. "
                f"Raw unscaled features will be used — accuracy may suffer."
            )
            return

        try:
            import joblib

            t_start = time.perf_counter_ns()
            self._scaler = joblib.load(self.scaler_path)
            log_latency(logger, "SCALER_LOAD", t_start)
            logger.info("✅ Feature scaler loaded successfully")

        except Exception as e:
            logger.exception(f"Failed to load scaler: {e}")

    # ═════════════════════════════════════════════════════════════════════
    #  PREDICTION
    # ═════════════════════════════════════════════════════════════════════

    def predict(self, features: np.ndarray) -> dict:
        """
        Run inference on a feature vector and return a trading signal.

        Args:
            features: 1-D float32 numpy array of shape (n_features,).

        Returns:
            {
                "signal":     int   — +1 (BUY), -1 (SELL), 0 (HOLD)
                "confidence": float — model probability (0.0–1.0)
                "latency_ms": float — end-to-end inference latency
                "raw_output": any   — raw ONNX output for debugging
            }
        """
        t_start = time.perf_counter_ns()

        # ── No model loaded → HOLD (dry run only) ────────────────────────
        if self._session is None:
            return self._make_result(self.SIGNAL_HOLD, 0.0, t_start, None)

        try:
            # ── Validate feature dimensionality ──────────────────────────
            if (
                self._expected_features is not None
                and features.shape[0] != self._expected_features
            ):
                logger.error(
                    f"Feature dim mismatch: got {features.shape[0]}, "
                    f"expected {self._expected_features}. Returning HOLD."
                )
                return self._make_result(self.SIGNAL_HOLD, 0.0, t_start, None)

            # ── Scale features ───────────────────────────────────────────
            x = features.reshape(1, -1).astype(np.float32)
            if self._scaler is not None:
                x = self._scaler.transform(x).astype(np.float32)

            # ── ONNX inference ───────────────────────────────────────────
            raw = self._session.run(
                [self._output_name],
                {self._input_name: x},
            )

            # ── Decode output ────────────────────────────────────────────
            signal, confidence = self._decode_output(raw)

            # ── Apply minimum confidence gate ────────────────────────────
            if signal != self.SIGNAL_HOLD and confidence < self.min_confidence:
                logger.debug(
                    f"Signal {signal} suppressed: confidence {confidence:.3f} "
                    f"< min {self.min_confidence:.3f}"
                )
                signal = self.SIGNAL_HOLD

            return self._make_result(signal, confidence, t_start, raw)

        except Exception as e:
            logger.exception(f"Inference error: {e}")
            return self._make_result(self.SIGNAL_HOLD, 0.0, t_start, None)

    # ═════════════════════════════════════════════════════════════════════
    #  OUTPUT DECODING
    # ═════════════════════════════════════════════════════════════════════

    def _decode_output(
        self, raw_output: list
    ) -> tuple[int, float]:
        """
        Convert raw ONNX output to (signal, confidence).

        Supported output shapes:
        • (1, 3)  — 3-class softmax [P(sell), P(hold), P(buy)]
        • (1, 2)  — binary [P(hold/sell), P(buy)]
        • (1, 1) or (1,) — regression scalar

        Thresholds (buy_threshold, sell_threshold) are used consistently
        in all branches.
        """
        output = raw_output[0]

        # ── 3-class classification ───────────────────────────────────────
        if output.ndim == 2 and output.shape[1] == 3:
            probs = output[0]                         # [P(sell), P(hold), P(buy)]
            cls = int(np.argmax(probs))
            confidence = float(probs[cls])
            signal_map = {0: self.SIGNAL_SELL, 1: self.SIGNAL_HOLD, 2: self.SIGNAL_BUY}
            return signal_map[cls], confidence

        # ── 2-class classification ───────────────────────────────────────
        if output.ndim == 2 and output.shape[1] == 2:
            probs = output[0]
            buy_prob = float(probs[1])
            if buy_prob >= self.buy_threshold:
                return self.SIGNAL_BUY, buy_prob
            if buy_prob <= self.sell_threshold:
                return self.SIGNAL_SELL, 1.0 - buy_prob
            # Dead zone between thresholds → HOLD
            return self.SIGNAL_HOLD, max(buy_prob, 1.0 - buy_prob)

        # ── Regression (single scalar) ───────────────────────────────────
        if output.size == 1:
            value = float(output.flat[0])
            confidence = min(abs(value), 1.0)
            if value >= self.buy_threshold:
                return self.SIGNAL_BUY, confidence
            if value <= -self.buy_threshold:
                return self.SIGNAL_SELL, confidence
            return self.SIGNAL_HOLD, 1.0 - confidence

        # ── Unknown shape ────────────────────────────────────────────────
        logger.warning(f"Unknown ONNX output shape: {output.shape}. Returning HOLD.")
        return self.SIGNAL_HOLD, 0.0

    # ═════════════════════════════════════════════════════════════════════
    #  HELPERS
    # ═════════════════════════════════════════════════════════════════════

    def _make_result(
        self,
        signal: int,
        confidence: float,
        t_start: int,
        raw_output,
    ) -> dict:
        """Measure latency, update counters, build result dict."""
        elapsed_ns = time.perf_counter_ns() - t_start
        elapsed_ms = elapsed_ns / 1_000_000

        with self._lock:
            self._total_inferences += 1
            self._total_latency_ns += elapsed_ns
            self._latency_ring.append(elapsed_ns)
            self._signal_counts[signal] = self._signal_counts.get(signal, 0) + 1

        if self._total_inferences % 100 == 0:
            avg_ms = (self._total_latency_ns / self._total_inferences) / 1_000_000
            p95 = np.percentile(list(self._latency_ring), 95) / 1_000_000
            logger.info(
                f"📊 Inference #{self._total_inferences} | "
                f"Avg: {avg_ms:.3f}ms | p95: {p95:.3f}ms | "
                f"This: {elapsed_ms:.3f}ms | Signal: {signal} ({confidence:.2%})"
            )

        return {
            "signal": signal,
            "confidence": confidence,
            "latency_ms": elapsed_ms,
            "raw_output": raw_output,
        }

    # ═════════════════════════════════════════════════════════════════════
    #  DIAGNOSTICS
    # ═════════════════════════════════════════════════════════════════════

    @property
    def is_ready(self) -> bool:
        """True when model is loaded and ready."""
        return self._session is not None

    @property
    def stats(self) -> dict:
        """Return performance and signal distribution statistics."""
        with self._lock:
            n = self._total_inferences
            avg_ms = (self._total_latency_ns / n / 1_000_000) if n > 0 else 0.0
            p95_ms = (
                float(np.percentile(list(self._latency_ring), 95)) / 1_000_000
                if self._latency_ring else 0.0
            )
            return {
                "total_inferences": n,
                "avg_latency_ms": round(avg_ms, 4),
                "p95_latency_ms": round(p95_ms, 4),
                "model_loaded": self._session is not None,
                "scaler_loaded": self._scaler is not None,
                "signal_distribution": dict(self._signal_counts),
            }
