
"""Solution for VisioMel Challenge"""

from joblib import load
from pathlib import Path

from loguru import logger
import numpy as np
import pandas as pd


DATA_ROOT = Path("/code_execution/data/")


def preprocess_feats(df, enc):
    df['age'] = df['age'].str.slice(1, 3).astype(np.float32)
    df['age'] = (df['age'] - df['age'].min()) / (df['age'].max() - df['age'].min())
    df['sex'] = df['sex'].replace(1, 0).replace(2, 1).astype(np.float32)
    df['melanoma_history'] = df['melanoma_history'].replace({'YES': 1, "NO": 0}).fillna(-1).astype(np.float32)
    df = pd.concat([df, pd.get_dummies(df['body_site']).astype(int)], axis=1).drop('body_site', axis=1)
    return df


def main():
    # load sumission format
    submission_format = pd.read_csv(DATA_ROOT / "submission_format.csv", index_col=0)

    # load test_metadata
    test_metadata = pd.read_csv(DATA_ROOT / "test_metadata.csv", index_col=0)

    logger.info("Loading feature encoder and model")
    calibrated_rf = load("assets/random_forest_model.joblib")
    history_encoder = load("assets/history_encoder.joblib")

    logger.info("Preprocessing features")
    processed_features = preprocess_feats(test_metadata, history_encoder)

    logger.info("Checking test feature filenames are in the same order as the submission format")
    assert (processed_features.index == submission_format.index).all()

    logger.info("Checking test feature columns align with loaded model")
    assert (processed_features.columns == calibrated_rf.feature_names_in_).all()

    logger.info("Generating predictions")
    submission_format["relapse"] = calibrated_rf.predict_proba(processed_features)[:,1]

    # save as "submission.csv" in the root folder, where it is expected
    logger.info("Writing out submission.csv")
    submission_format.to_csv("submission.csv")


if __name__ == "__main__":
    main()
