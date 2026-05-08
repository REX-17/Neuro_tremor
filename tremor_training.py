"""
tremor_training.py
------------------
Trains a neural network on CSV tremor data, exports:
  1. tremor_model.tflite  → used by ml_model.py for inference
  2. model.h              → C header for embedding in ESP32 firmware

CSV format expected (header row required):
  tremor_pct, hr_bpm, combined_pct, label
  
  label values: 0=Normal, 1=Mild, 2=Moderate, 3=Severe

Usage:
  python tremor_training.py                      # uses tremor_data.csv
  python tremor_training.py --csv my_data.csv    # custom CSV path

If no CSV exists, a synthetic dataset is generated for demonstration.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s', datefmt='%H:%M:%S')
log = logging.getLogger(__name__)

CSV_PATH        = "tremor_data.csv"
TFLITE_OUT      = "tremor_model.tflite"
HEADER_OUT      = "model.h"
EPOCHS          = 80
BATCH_SIZE      = 16
TEST_SPLIT      = 0.2
RANDOM_SEED     = 42


# ── Synthetic dataset generator ──────────────────────────────
def generate_synthetic_dataset(n=800) -> pd.DataFrame:
    """
    Creates a labelled dataset by simulating sensor readings
    for 4 severity classes.
    """
    np.random.seed(RANDOM_SEED)
    rows = []

    class_configs = [
        # (label, tremor_mean, tremor_std, hr_mean, hr_std, n_samples)
        (0, 2.0,  3.0,  72.0,  6.0,  n // 4),   # Normal
        (1, 18.0, 8.0,  78.0,  8.0,  n // 4),   # Mild
        (2, 45.0, 12.0, 88.0,  10.0, n // 4),   # Moderate
        (3, 75.0, 15.0, 100.0, 14.0, n // 4),   # Severe
    ]

    for label, t_mean, t_std, hr_mean, hr_std, count in class_configs:
        for _ in range(count):
            t   = float(np.clip(np.random.normal(t_mean,  t_std),  0, 100))
            hr  = float(np.clip(np.random.normal(hr_mean, hr_std), 30, 180))

            # HR deviation contribution (mirrors Arduino logic)
            if 55 <= hr <= 90:
                hr_sev = 0.0
            elif hr < 55:
                hr_sev = np.clip((55 - hr) / 20.0 * 100, 0, 100)
            else:
                hr_sev = np.clip((hr - 90) / 20.0 * 100, 0, 100)

            combined = (t * 0.70) + (hr_sev * 0.30)
            rows.append({
                "tremor_pct":   round(t, 2),
                "hr_bpm":       round(hr, 1),
                "combined_pct": round(combined, 2),
                "label":        label,
            })

    df = pd.DataFrame(rows).sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    log.info(f"Generated synthetic dataset: {len(df)} samples")
    log.info(f"Class distribution:\n{df['label'].value_counts().sort_index()}")
    return df


# ── Feature engineering ───────────────────────────────────────
def build_features(df: pd.DataFrame) -> np.ndarray:
    t_n   = np.clip(df["tremor_pct"].values   / 100.0, 0, 1)
    hr_n  = np.clip(df["hr_bpm"].values       / 200.0, 0, 1)
    c_n   = np.clip(df["combined_pct"].values / 100.0, 0, 1)
    hr_dev = np.clip(np.abs(df["hr_bpm"].values - 72.0) / 50.0, 0, 1)
    inter  = t_n * hr_dev
    return np.column_stack([t_n, hr_n, c_n, hr_dev, inter]).astype(np.float32)


# ── Model definition and training ────────────────────────────
def train(csv_path: str):
    try:
        import tensorflow as tf
        from tensorflow import keras
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import classification_report
    except ImportError as e:
        log.error(f"Missing dependency: {e}")
        log.error("Run: pip install tensorflow scikit-learn pandas numpy")
        sys.exit(1)

    # ── Load or generate data ────────────────────────────────
    if os.path.exists(csv_path):
        log.info(f"Loading dataset from {csv_path}")
        df = pd.read_csv(csv_path)
        required = {"tremor_pct", "hr_bpm", "combined_pct", "label"}
        if not required.issubset(df.columns):
            log.error(f"CSV must have columns: {required}")
            sys.exit(1)
    else:
        log.warning(f"'{csv_path}' not found. Generating synthetic dataset...")
        df = generate_synthetic_dataset(n=800)
        df.to_csv(csv_path, index=False)
        log.info(f"Synthetic dataset saved to {csv_path}")

    X = build_features(df)
    y = df["label"].values.astype(np.int32)
    num_classes = len(np.unique(y))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SPLIT, random_state=RANDOM_SEED, stratify=y
    )
    log.info(f"Train: {len(X_train)} | Test: {len(X_test)} | Features: {X.shape[1]} | Classes: {num_classes}")

    # ── Build model ──────────────────────────────────────────
    model = keras.Sequential([
        keras.layers.Input(shape=(X.shape[1],)),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(16, activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(8, activation='relu'),
        keras.layers.Dense(num_classes, activation='softmax'),
    ], name="tremor_classifier")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    model.summary()

    # ── Train ────────────────────────────────────────────────
    callbacks = [
        keras.callbacks.EarlyStopping(patience=12, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=6, min_lr=1e-5),
    ]
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1,
    )

    # ── Evaluate ─────────────────────────────────────────────
    _, acc = model.evaluate(X_test, y_test, verbose=0)
    log.info(f"\nTest accuracy: {acc * 100:.1f}%")
    y_pred = np.argmax(model.predict(X_test), axis=1)
    labels = ["Normal", "Mild", "Moderate", "Severe"]
    print("\n" + classification_report(y_test, y_pred, target_names=labels))

    # ── Convert to TFLite ────────────────────────────────────
    log.info("Converting to TFLite...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]   # quantise for smaller size
    tflite_model = converter.convert()

    with open(TFLITE_OUT, "wb") as f:
        f.write(tflite_model)
    log.info(f"TFLite model saved → {TFLITE_OUT} ({len(tflite_model):,} bytes)")

    # ── Convert to C header for ESP32 ────────────────────────
    log.info(f"Generating C header → {HEADER_OUT}")
    hex_array = ", ".join(f"0x{b:02x}" for b in tflite_model)
    header_content = f"""\
/*
 * model.h — Auto-generated TFLite model for ESP32
 * Generated by tremor_training.py
 * Model size: {len(tflite_model):,} bytes
 * Accuracy:   {acc * 100:.1f}%
 *
 * Usage in Arduino sketch:
 *   #include "model.h"
 *   const tflite::Model* model = tflite::GetModel(g_tremor_model);
 */

#ifndef TREMOR_MODEL_H
#define TREMOR_MODEL_H

#include <stdint.h>

// Model data array
alignas(8) const uint8_t g_tremor_model[] = {{
  {hex_array}
}};

const int g_tremor_model_len = {len(tflite_model)};

#endif  // TREMOR_MODEL_H
"""
    with open(HEADER_OUT, "w") as f:
        f.write(header_content)
    log.info(f"C header saved → {HEADER_OUT}")
    log.info("\nDone! Files generated:")
    log.info(f"  {TFLITE_OUT}  → used by ml_model.py")
    log.info(f"  {HEADER_OUT}          → copy to your Arduino sketch folder")


# ── Entry point ───────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Parkinson's tremor classifier")
    parser.add_argument("--csv", default=CSV_PATH, help=f"CSV dataset path (default: {CSV_PATH})")
    args = parser.parse_args()
    train(args.csv)