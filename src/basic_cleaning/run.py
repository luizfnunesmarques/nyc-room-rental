#!/usr/bin/env python
"""
An example of a step using MLflow and Weights & Biases
"""
import argparse
import logging
import wandb
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

def go(args):
    logger.info("Initialising basic cleaning step.")

    run = wandb.init(project="nyc_airbnb", group="eda", save_code=True, job_type="basic_cleaning")
    run.config.update(args)

    local_path = wandb.use_artifact(args.input_artifact).file()
    df = pd.read_csv(local_path)

    logger.info("File read.")

    min_price = args.min_price
    max_price = args.max_price

    logger.info("Removing outliers.")
    normalised_data = df['price'].between(min_price, max_price)
    df = df[normalised_data].copy()

    logger.info("Transforming date fields.")
    df['last_review'] = pd.to_datetime(df['last_review'])

    df.to_csv("clean_sample.csv", index=False)

    artifact = wandb.Artifact(
        args.output_artifact,
        type=args.output_type,
        description=args.output_description,
    )

    artifact.add_file("clean_sample.csv")
    run.log_artifact(artifact)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data cleaning step")

    parser.add_argument(
        "--input_artifact",
        type=str,
        help="Input artifact to be cleaned.",
        required=True
    )

    parser.add_argument(
        "--output_artifact",
        type=str,
        help="Cleaned data.",
        required=True
    )

    parser.add_argument(
        "--output_type",
        type=str,
        help="String",
        required=True
    )

    parser.add_argument(
        "--output_description",
        type=str,
        help="",
        required=True
    )

    parser.add_argument(
        "--min_price",
        type=float,
        help="",
        required=True
    )

    parser.add_argument(
        "--max_price",
        type=float,
        help="",
        required=True
    )


    args = parser.parse_args()

    go(args)
