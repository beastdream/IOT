"""Portable paths for this source checkout; importing creates no files."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_CONFIG = PROJECT_ROOT / "configs" / "data.local.yaml"
ORIGINAL_DATA_CONFIG = PROJECT_ROOT / "data.yaml"
TRAIN_IMAGES_DIR = PROJECT_ROOT / "train" / "images"
TRAIN_LABELS_DIR = PROJECT_ROOT / "train" / "labels"
VALID_IMAGES_DIR = PROJECT_ROOT / "valid" / "images"
VALID_LABELS_DIR = PROJECT_ROOT / "valid" / "labels"
TEST_IMAGES_DIR = PROJECT_ROOT / "test" / "images"
TEST_LABELS_DIR = PROJECT_ROOT / "test" / "labels"
RESULTS_DIR = PROJECT_ROOT / "results"
DATASET_AUDIT_DIR = RESULTS_DIR / "dataset_audit"
DATASET_ANALYSIS_DIR = RESULTS_DIR / "dataset_analysis"
DATASET_CLEANING_DIR = RESULTS_DIR / "dataset_cleaning"
EXPERIMENTS_DIR = RESULTS_DIR / "experiments"
EVALUATION_DIR = RESULTS_DIR / "evaluation"
PREDICTIONS_DIR = RESULTS_DIR / "predictions"
FAILURE_ANALYSIS_DIR = RESULTS_DIR / "failure_analysis"
