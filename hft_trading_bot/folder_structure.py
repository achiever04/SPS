"""
===============================================================================
  HFT Trading Bot — Automated Project Structure Generator
===============================================================================
  Run this script ONCE to generate the complete directory tree, placeholder
  files, and foundational config files (.env, .gitignore, requirements.txt).

  Usage:
      python folder_structure.py

  This script is idempotent — running it multiple times will not overwrite
  existing files (uses exist_ok=True for directories).
===============================================================================
"""

import os
from pathlib import Path


def create_project_structure():
    """Generate the full HFT Trading Bot project directory tree."""

    # ── Root of the project (same directory as this script) ──────────────
    BASE_DIR = Path(__file__).resolve().parent

    # ── 1. Define all directories to create ──────────────────────────────
    directories = [
        "config",
        "core",
        "models",
        "logs",
        "utils",
        "tests",
    ]

    # ── 2. Define files with their default content ───────────────────────
    files = {
        # --- Package init files -------------------------------------------
        "config/__init__.py": '"""Configuration package."""\n',
        "core/__init__.py": '"""Core trading engine modules."""\n',
        "utils/__init__.py": '"""Utility modules."""\n',
        "tests/__init__.py": '"""Test suite."""\n',

        # --- Placeholder .gitkeep for empty dirs --------------------------
        "models/.gitkeep": "",
        "logs/.gitkeep": "",

        # --- Environment variables template -------------------------------
        ".env": (
            "# ============================================================\n"
            "#  HFT Trading Bot — Environment Variables\n"
            "# ============================================================\n"
            "#  IMPORTANT: Never commit this file to version control.\n"
            "# ============================================================\n"
            "\n"
            "# Broker API Credentials\n"
            "BROKER_API_KEY=your_api_key_here\n"
            "BROKER_API_SECRET=your_api_secret_here\n"
            "BROKER_ACCESS_TOKEN=your_access_token_here\n"
            "\n"
            "# WebSocket Configuration\n"
            'WEBSOCKET_URL=wss://your-broker-ws-endpoint.com/ws\n'
            "\n"
            "# Trading Parameters\n"
            "TRADE_SYMBOL=NIFTY\n"
            "TRADE_EXCHANGE=NFO\n"
            "MAX_POSITION_SIZE=50\n"
            "STOP_LOSS_PERCENT=0.5\n"
            "MAX_OPEN_POSITIONS=3\n"
            "\n"
            "# Risk Management\n"
            "MAX_DAILY_LOSS=10000\n"
            "MAX_ORDER_VALUE=500000\n"
            "\n"
            "# Logging\n"
            'LOG_LEVEL=INFO\n'
            'LOG_FILE=logs/trading_bot.log\n'
        ),

        # --- Git Ignore ---------------------------------------------------
        ".gitignore": (
            "# ============================================================\n"
            "#  HFT Trading Bot — .gitignore\n"
            "# ============================================================\n"
            "\n"
            "# Environment & Secrets\n"
            ".env\n"
            ".env.*\n"
            "\n"
            "# Python\n"
            "__pycache__/\n"
            "*.py[cod]\n"
            "*$py.class\n"
            "*.egg-info/\n"
            "dist/\n"
            "build/\n"
            "*.egg\n"
            "\n"
            "# Virtual Environment\n"
            "venv/\n"
            ".venv/\n"
            "env/\n"
            "\n"
            "# Models (large binary files)\n"
            "models/*.onnx\n"
            "models/*.pkl\n"
            "models/*.joblib\n"
            "\n"
            "# Logs\n"
            "logs/*.log\n"
            "logs/*.log.*\n"
            "\n"
            "# IDE\n"
            ".vscode/\n"
            ".idea/\n"
            "*.swp\n"
            "*.swo\n"
            "\n"
            "# OS\n"
            ".DS_Store\n"
            "Thumbs.db\n"
        ),

        # --- Requirements -------------------------------------------------
        "requirements.txt": (
            "# ============================================================\n"
            "#  HFT Trading Bot — Production Dependencies\n"
            "# ============================================================\n"
            "#  Install: pip install -r requirements.txt\n"
            "# ============================================================\n"
            "\n"
            "# Async HTTP & WebSocket\n"
            "aiohttp>=3.9.0\n"
            "\n"
            "# ONNX Runtime (CPU-only for minimal footprint)\n"
            "onnxruntime>=1.17.0\n"
            "\n"
            "# Numerical computation (NO Pandas in live pipeline)\n"
            "numpy>=1.26.0\n"
            "\n"
            "# Environment variable management\n"
            "python-dotenv>=1.0.0\n"
            "\n"
            "# Serialization (for scaler loading)\n"
            "joblib>=1.3.0\n"
            "\n"
            "# Testing\n"
            "pytest>=7.4.0\n"
            "pytest-asyncio>=0.23.0\n"
        ),
    }

    # ── 3. Create directories ────────────────────────────────────────────
    print("=" * 60)
    print("  HFT Trading Bot — Project Structure Generator")
    print("=" * 60)
    print()

    for dir_name in directories:
        dir_path = BASE_DIR / dir_name
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"  [DIR]  Created: {dir_name}/")

    print()

    # ── 4. Create files (skip if already exists) ─────────────────────────
    for file_rel_path, content in files.items():
        file_path = BASE_DIR / file_rel_path
        if not file_path.exists():
            file_path.write_text(content, encoding="utf-8")
            print(f"  [FILE] Created: {file_rel_path}")
        else:
            print(f"  [SKIP] Already exists: {file_rel_path}")

    # ── 5. Print the final tree ──────────────────────────────────────────
    print()
    print("=" * 60)
    print("  Project structure generated successfully!")
    print("=" * 60)
    print()
    print("  hft_trading_bot/")
    print("  ├── main.py")
    print("  ├── folder_structure.py")
    print("  ├── requirements.txt")
    print("  ├── .env")
    print("  ├── .gitignore")
    print("  │")
    print("  ├── config/")
    print("  │   ├── __init__.py")
    print("  │   └── config.py")
    print("  │")
    print("  ├── core/")
    print("  │   ├── __init__.py")
    print("  │   ├── websocket_client.py")
    print("  │   ├── feature_engine.py")
    print("  │   ├── inference_engine.py")
    print("  │   └── execution_engine.py")
    print("  │")
    print("  ├── models/")
    print("  │   └── (.gitkeep)")
    print("  │")
    print("  ├── logs/")
    print("  │   └── (.gitkeep)")
    print("  │")
    print("  ├── utils/")
    print("  │   ├── __init__.py")
    print("  │   └── logger.py")
    print("  │")
    print("  └── tests/")
    print("      ├── __init__.py")
    print("      └── test_feature_engine.py")
    print()


if __name__ == "__main__":
    create_project_structure()
