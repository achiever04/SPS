"""
===============================================================================
  train_model.py — Train & Export a Trading Signal Model to ONNX
===============================================================================
  This script:
  1. Generates synthetic training data (replace with your real historical data)
  2. Computes the same features the live bot uses (RSI, MACD, Bollinger Bands)
  3. Trains a lightweight classifier (RandomForest or GradientBoosting)
  4. Exports the model to ONNX format → models/model.onnx
  5. Saves the fitted scaler → models/scaler.pkl

  After running this, the live bot will load model.onnx and scaler.pkl
  automatically from the models/ directory.

  Usage:
      pip install scikit-learn skl2onnx
      python scripts/train_model.py

  For REAL trading, replace the synthetic data section with your actual
  historical OHLCV data from your broker or data provider.
===============================================================================
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import joblib

warnings.filterwarnings("ignore")

# ── Ensure project root is in sys.path ───────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def generate_synthetic_data(n_samples: int = 10000) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic price data and compute features.
    
    ⚠️ REPLACE THIS with real historical OHLCV data for production use.
    
    Data sources for real Indian market data:
    • Zerodha Kite Historical API: https://kite.trade/docs/connect/v3/historical/
    • Yahoo Finance: pip install yfinance → yf.download("^NSEI")
    • NSE India: https://www.nseindia.com/
    • Angel One Historical API
    
    Returns:
        features: (n_samples, 9) array of computed indicators
        labels:   (n_samples,) array of {0: SELL, 1: HOLD, 2: BUY}
    """
    from core.feature_engine import FeatureEngine
    
    print("📊 Generating synthetic training data...")
    print("   ⚠️  Replace this with real historical data for production!")
    
    np.random.seed(42)
    
    # ── Simulate a random walk price series ──────────────────────────────
    # Starting price ~19,500 (NIFTY-like)
    returns = np.random.normal(0.0001, 0.005, n_samples + 200)
    prices = 19500 * np.cumprod(1 + returns)
    
    # ── Compute features using the same FeatureEngine the live bot uses ──
    engine = FeatureEngine(
        rsi_period=14,
        macd_fast=12,
        macd_slow=26,
        macd_signal=9,
        bollinger_period=20,
        bollinger_std_dev=2.0,
    )
    
    features_list = []
    price_list = []
    
    for price in prices:
        result = engine.update(price)
        if result is not None:
            features_list.append(result.copy())
            price_list.append(price)
    
    features = np.array(features_list, dtype=np.float32)
    corresponding_prices = np.array(price_list)
    
    # ── Generate labels based on future returns ──────────────────────────
    # Look 5 bars ahead: if price goes up > 0.1%, label = BUY (2)
    #                     if price goes down > 0.1%, label = SELL (0)
    #                     otherwise, label = HOLD (1)
    lookahead = 5
    labels = np.ones(len(features), dtype=np.int64)  # Default: HOLD
    
    for i in range(len(features) - lookahead):
        future_return = (corresponding_prices[i + lookahead] - corresponding_prices[i]) / corresponding_prices[i]
        if future_return > 0.001:    # +0.1%
            labels[i] = 2  # BUY
        elif future_return < -0.001:  # -0.1%
            labels[i] = 0  # SELL
        else:
            labels[i] = 1  # HOLD
    
    # Trim the last `lookahead` samples (no future data available)
    features = features[:-lookahead]
    labels = labels[:-lookahead]
    
    print(f"   ✅ Generated {len(features)} samples with {features.shape[1]} features")
    print(f"   📊 Label distribution: SELL={np.sum(labels==0)}, "
          f"HOLD={np.sum(labels==1)}, BUY={np.sum(labels==2)}")
    
    return features, labels


def train_and_export():
    """Train a model, export to ONNX, and save the scaler."""
    
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import classification_report
    
    # ── Step 1: Generate / Load Data ─────────────────────────────────────
    features, labels = generate_synthetic_data(n_samples=10000)
    
    # ── Step 2: Train/Test Split ─────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42, stratify=labels
    )
    print(f"\n📐 Train: {len(X_train)} samples | Test: {len(X_test)} samples")
    
    # ── Step 3: Scale Features ───────────────────────────────────────────
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # ── Step 4: Train Model ──────────────────────────────────────────────
    print("\n🏋️ Training GradientBoosting classifier...")
    model = GradientBoostingClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        random_state=42,
    )
    model.fit(X_train_scaled, y_train)
    
    # ── Step 5: Evaluate ─────────────────────────────────────────────────
    y_pred = model.predict(X_test_scaled)
    print("\n📋 Classification Report:")
    print(classification_report(
        y_test, y_pred, 
        target_names=["SELL", "HOLD", "BUY"]
    ))
    
    accuracy = np.mean(y_pred == y_test)
    print(f"   Accuracy: {accuracy:.2%}")
    
    # ── Step 6: Export to ONNX ───────────────────────────────────────────
    print("\n📦 Exporting model to ONNX format...")
    
    try:
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType
        
        # Define the input shape: (batch_size, n_features)
        initial_type = [("input", FloatTensorType([None, X_train.shape[1]]))]
        
        # Convert to ONNX
        onnx_model = convert_sklearn(
            model, 
            initial_types=initial_type,
            target_opset=13,
            options={type(model): {"zipmap": False}},  # Return array, not dict
        )
        
        # Save the ONNX model
        model_path = PROJECT_ROOT / "models" / "model.onnx"
        with open(model_path, "wb") as f:
            f.write(onnx_model.SerializeToString())
        
        print(f"   ✅ Model saved to: {model_path}")
        print(f"   📏 Model size: {model_path.stat().st_size / 1024:.1f} KB")
        
    except ImportError:
        print("   ❌ skl2onnx not installed! Install with:")
        print("      pip install skl2onnx")
        print("   Skipping ONNX export.")
        return
    
    # ── Step 7: Save the Scaler ──────────────────────────────────────────
    scaler_path = PROJECT_ROOT / "models" / "scaler.pkl"
    joblib.dump(scaler, scaler_path)
    print(f"   ✅ Scaler saved to: {scaler_path}")
    
    # ── Step 8: Verify ONNX Model ────────────────────────────────────────
    print("\n🔍 Verifying ONNX model...")
    import onnxruntime as ort
    
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    
    # Test with a single sample
    test_input = scaler.transform(X_test[:1])
    test_input = test_input.astype(np.float32)
    
    result = session.run([output_name], {input_name: test_input})
    print(f"   Input shape:  {test_input.shape}")
    print(f"   Output shape: {result[0].shape}")
    print(f"   Prediction:   {result[0]}")
    print(f"   ✅ ONNX model verified — ready for live inference!")
    
    # ── Summary ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  🎉 TRAINING COMPLETE")
    print("=" * 60)
    print(f"  Model:  models/model.onnx ({model_path.stat().st_size / 1024:.1f} KB)")
    print(f"  Scaler: models/scaler.pkl")
    print(f"  Accuracy: {accuracy:.2%}")
    print()
    print("  The live bot will now auto-load these files on startup.")
    print("  Run: python main.py")
    print("=" * 60)


if __name__ == "__main__":
    train_and_export()
