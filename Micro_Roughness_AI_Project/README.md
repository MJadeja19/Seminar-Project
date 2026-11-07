# Micro Component Surface Roughness AI/ML Project

## What this project does
Trains an Extra Trees regression model using the supplied Excel dataset to predict:
- Ra (micrometres)
- Rq (micrometres)
- Rz (micrometres)

It uses seven machining inputs:
1. Tool diameter (mm)
2. Spindle speed (RPM)
3. Feed rate (mm/min)
4. Feed per tooth (µm)
5. Depth of cut (µm)
6. Width of cut (µm)
7. Tool wear (µm)

## Setup in PyCharm
1. Create/open a PyCharm project.
2. Copy all files from this folder into the project.
3. Put `Micro_Component_Roughness_Dataset_10000.xlsx` in the same folder.
4. Open PyCharm Terminal and run:

```bash
pip install -r requirements.txt
```

5. Train the model:

```bash
python train_model.py
```

6. Predict roughness:

```bash
python predict_roughness.py
```

7. Enter the seven parameters when prompted. The program prints Ra, Rq and Rz.

## Important dataset limitation
The workbook's `Notes_Method` sheet says the 10,000 rows are simulated/synthetic,
not measurements collected from a real machining experiment. The model is useful
for prototyping and learning the ML workflow. Do not describe its scores as
experimental validation or use predictions as certified engineering values.
For a research-quality model, collect actual machining inputs paired with
profilometer-measured roughness values, then retrain and validate on those data.

## Model evaluation
`train_model.py` holds out 20% of the rows for testing and reports R², MAE and
RMSE separately for Ra, Rq and Rz. It saves:
- `roughness_model.joblib`
- `model_info.json`
