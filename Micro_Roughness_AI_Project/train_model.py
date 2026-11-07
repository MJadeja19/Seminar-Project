"""
Train a multi-output ML model to predict Ra, Rq and Rz.
Dataset: Micro_Component_Roughness_Dataset_10000.xlsx
Run: python train_model.py
"""

from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "Micro_Component_Roughness_Dataset_10000.xlsx"
SHEET_NAME = "Micro_Component_Data"
MODEL_FILE = BASE_DIR / "roughness_model.joblib"
INFO_FILE = BASE_DIR / "model_info.json"

FEATURES = [
    "Tool_Diameter_mm",
    "Spindle_Speed_RPM",
    "Feed_Rate_mm_min",
    "Feed_per_Tooth_um",
    "Depth_of_Cut_um",
    "Width_of_Cut_um",
    "Tool_Wear_um",
]

TARGETS = [
    "Surface_Roughness_Ra_um",
    "Rq_um",
    "Rz_um",
]

DISPLAY_NAMES = ["Ra", "Rq", "Rz"]


def main():
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_FILE}\n"
            "Place the Excel file in the same folder as train_model.py."
        )

    df = pd.read_excel(DATA_FILE, sheet_name=SHEET_NAME)

    required = FEATURES + TARGETS
    missing_columns = [c for c in required if c not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing Excel columns: {missing_columns}")

    # Keep only the selected model inputs and target outputs.
    data = df[required].copy()
    data = data.replace([np.inf, -np.inf], np.nan).dropna()

    X = data[FEATURES]
    y = data[TARGETS]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    # Extra Trees works well for nonlinear relationships in tabular data.
    model = ExtraTreesRegressor(
        n_estimators=400,
        min_samples_leaf=2,
        max_features=1.0,
        n_jobs=-1,
        random_state=42,
    )

    print(f"Training with {len(X_train):,} rows...")
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    print("\n========== TEST SET PERFORMANCE ==========")
    r2_values = r2_score(y_test, predictions, multioutput="raw_values")
    mae_values = mean_absolute_error(y_test, predictions, multioutput="raw_values")
    rmse_values = np.sqrt(
        mean_squared_error(y_test, predictions, multioutput="raw_values")
    )

    for name, r2, mae, rmse in zip(
        DISPLAY_NAMES, r2_values, mae_values, rmse_values
    ):
        print(f"{name}: R² = {r2:.4f} | MAE = {mae:.4f} µm | RMSE = {rmse:.4f} µm")

    joblib.dump(model, MODEL_FILE)

    info = {
        "features": FEATURES,
        "targets": TARGETS,
        "display_names": DISPLAY_NAMES,
        "training_rows": int(len(X_train)),
        "testing_rows": int(len(X_test)),
        "dataset_rows_used": int(len(data)),
        "feature_ranges": {
            feature: {
                "min": float(X[feature].min()),
                "max": float(X[feature].max()),
            }
            for feature in FEATURES
        },
        "test_metrics": {
            name: {
                "r2": float(r2),
                "mae_um": float(mae),
                "rmse_um": float(rmse),
            }
            for name, r2, mae, rmse in zip(
                DISPLAY_NAMES, r2_values, mae_values, rmse_values
            )
        },
        "dataset_note": (
            "The supplied workbook identifies these 10,000 rows as synthetic, "
            "not real experimental measurements."
        ),
    }

    INFO_FILE.write_text(json.dumps(info, indent=4), encoding="utf-8")

    print(f"\nModel saved: {MODEL_FILE.name}")
    print(f"Model information saved: {INFO_FILE.name}")
    print("\nTraining completed successfully.")


if __name__ == "__main__":
    main()
