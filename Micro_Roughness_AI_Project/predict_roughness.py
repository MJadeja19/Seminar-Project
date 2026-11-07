"""
Enter machining parameters and predict Ra, Rq and Rz.
Run after train_model.py:
    python predict_roughness.py
"""

from pathlib import Path
import json
import joblib
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
MODEL_FILE = BASE_DIR / "roughness_model.joblib"
INFO_FILE = BASE_DIR / "model_info.json"

if not MODEL_FILE.exists() or not INFO_FILE.exists():
    raise FileNotFoundError(
        "Trained model not found. First run: python train_model.py"
    )

model = joblib.load(MODEL_FILE)
info = json.loads(INFO_FILE.read_text(encoding="utf-8"))

FEATURES = info["features"]
RANGES = info["feature_ranges"]

PROMPTS = {
    "Tool_Diameter_mm": "Tool diameter (mm)",
    "Spindle_Speed_RPM": "Spindle speed (RPM)",
    "Feed_Rate_mm_min": "Feed rate (mm/min)",
    "Feed_per_Tooth_um": "Feed per tooth (micrometres)",
    "Depth_of_Cut_um": "Depth of cut (micrometres)",
    "Width_of_Cut_um": "Width of cut (micrometres)",
    "Tool_Wear_um": "Tool wear (micrometres)",
}


def get_number(feature):
    while True:
        try:
            value = float(input(f"{PROMPTS[feature]}: "))
            if value < 0:
                print("Please enter a non-negative value.")
                continue

            low = RANGES[feature]["min"]
            high = RANGES[feature]["max"]

            if not low <= value <= high:
                print(
                    f"WARNING: Dataset range for this parameter is "
                    f"{low:.3f} to {high:.3f}. Your value is outside it."
                )
                confirm = input("Continue anyway? (y/n): ").strip().lower()
                if confirm != "y":
                    continue

            return value
        except ValueError:
            print("Invalid input. Please enter a number.")


def main():
    print("\n===== MICRO-MACHINING SURFACE ROUGHNESS PREDICTOR =====")
    print("Enter the machining parameters below:\n")

    values = {feature: get_number(feature) for feature in FEATURES}
    input_df = pd.DataFrame([values], columns=FEATURES)

    result = model.predict(input_df)[0]

    print("\n========== PREDICTED SURFACE ROUGHNESS ==========")
    print(f"Ra = {result[0]:.4f} µm")
    print(f"Rq = {result[1]:.4f} µm")
    print(f"Rz = {result[2]:.4f} µm")
    print("\nNote: These are ML predictions from a model trained on synthetic data.")
    print("Validate against actual surface profilometer measurements before")
    print("using the predictions as engineering or experimental results.")


if __name__ == "__main__":
    main()
