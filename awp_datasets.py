"""Dataset metadata shared by AWP-MC training and result aggregation."""

from pathlib import Path


DATASETS = [
    "PDSetting1",
    "PDSetting2",
    "PronunciationAudio",
    "ECGFiveDays",
    "FreezerSmallTrain",
    "HouseTwenty",
    "InsectEPGRegularTrain",
    "ItalyPowerDemand",
    "Lightning7",
    "MoteStrain",
    "PowerCons",
    "SonyAIBORobotSurface2",
    "UWaveGestureLibraryAll",
]

# The released forecasting protocol excludes the Parkinson sensor settings.
# UWave is intentionally omitted from the retained local paper suite because
# it dominates runtime.
FORECAST_DATASETS = [
    "PronunciationAudio",
    "ECGFiveDays",
    "FreezerSmallTrain",
    "HouseTwenty",
    "InsectEPGRegularTrain",
    "ItalyPowerDemand",
    "Lightning7",
    "MoteStrain",
    "PowerCons",
    "SonyAIBORobotSurface2",
]

DATASET_PATHS = {
    "PDSetting1": Path("data/parkinson/PD setting 1.npz"),
    "PDSetting2": Path("data/parkinson/PD setting 2.npz"),
    "PronunciationAudio": Path("data/audio/Pronunciation Audio.npy"),
    "ECGFiveDays": Path("data/basics/ECGFiveDays.npy"),
    "FreezerSmallTrain": Path("data/basics/FreezerSmallTrain.npy"),
    "HouseTwenty": Path("data/basics/HouseTwenty.npy"),
    "InsectEPGRegularTrain": Path("data/basics/InsectEPGRegularTrain.npy"),
    "ItalyPowerDemand": Path("data/basics/ItalyPowerDemand.npy"),
    "Lightning7": Path("data/basics/Lightning7.npy"),
    "MoteStrain": Path("data/basics/MoteStrain.npy"),
    "PowerCons": Path("data/basics/PowerCons.npy"),
    "SonyAIBORobotSurface2": Path("data/basics/SonyAIBORobotSurface2.npy"),
    "UWaveGestureLibraryAll": Path("data/basics/UWaveGestureLibraryAll.npy"),
}

# Historical values reported by Bajaj and Nguyen. PD values come from the
# released repository table; the paper names PronunciationAudio as "Sound".
REPORTED_MOTION_CODE = {
    "PDSetting1": 71.12,
    "PDSetting2": 54.31,
    "PronunciationAudio": 87.50,
    "ECGFiveDays": 66.55,
    "FreezerSmallTrain": 70.25,
    "HouseTwenty": 70.59,
    "InsectEPGRegularTrain": 100.00,
    "ItalyPowerDemand": 72.50,
    "Lightning7": 31.51,
    "MoteStrain": 72.68,
    "PowerCons": 92.78,
    "SonyAIBORobotSurface2": 75.97,
    "UWaveGestureLibraryAll": 80.18,
}

# Exact RMSE values from the Motion Code forecasting table. The paper names
# PronunciationAudio as "Sound".
REPORTED_MOTION_CODE_FORECAST_RMSE = {
    "PronunciationAudio": 0.085,
    "ECGFiveDays": 0.27,
    "FreezerSmallTrain": 0.74,
    "HouseTwenty": 648.27,
    "InsectEPGRegularTrain": 0.048,
    "ItalyPowerDemand": 0.67,
    "Lightning7": 1.08,
    "MoteStrain": 0.82,
    "PowerCons": 1.15,
    "SonyAIBORobotSurface2": 2.26,
}
