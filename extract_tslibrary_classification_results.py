"""Extract retained TSLibrary classification accuracies for the paper tables."""

from __future__ import annotations

import csv
from pathlib import Path
import re
from statistics import mean


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
]

MODELS = [
    "Informer",
    "Autoformer",
    "FEDformer",
    "ETSformer",
    "LightTS",
    "PatchTST",
    "Crossformer",
    "DLinear",
    "TimesNet",
    "iTransformer",
    "Mamba",
]

RESULT_PATTERN = re.compile(r"^classification_(.+?)_([^_]+)_UEA_")
ACCURACY_PATTERN = re.compile(r"accuracy\s*:\s*([0-9.eE+-]+)")


def load_accuracies(root: Path) -> dict[tuple[str, str], float]:
    accuracies: dict[tuple[str, str], float] = {}
    for result_path in root.glob("*/result_classification.txt"):
        match = RESULT_PATTERN.match(result_path.parent.name)
        if match is None:
            continue
        accuracy_match = ACCURACY_PATTERN.search(
            result_path.read_text(encoding="utf-8", errors="replace")
        )
        if accuracy_match is None:
            raise RuntimeError(f"Missing accuracy in {result_path}")
        dataset, model = match.groups()
        accuracies[(dataset, model)] = 100.0 * float(accuracy_match.group(1))
    return accuracies


def main() -> None:
    accuracies = load_accuracies(Path("TSLibrary/results"))
    missing = [
        (dataset, model)
        for dataset in DATASETS
        for model in MODELS
        if (dataset, model) not in accuracies
    ]
    if missing:
        raise RuntimeError(f"Missing retained TSLibrary results: {missing}")

    output_path = Path("out/awp_paper_evidence/tslibrary_classification_baselines.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(["dataset", *MODELS])
        for dataset in DATASETS:
            writer.writerow(
                [dataset, *(f"{accuracies[(dataset, model)]:.8f}" for model in MODELS)]
            )
        writer.writerow(
            [
                "Macro average",
                *(
                    f"{mean(accuracies[(dataset, model)] for dataset in DATASETS):.8f}"
                    for model in MODELS
                ),
            ]
        )

    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
