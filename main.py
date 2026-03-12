from model_training import train_model


def run_cli():
    import argparse

    parser = argparse.ArgumentParser(description="Train barangay-specific flood prediction models.")
    parser.add_argument(
        "--force-retrain",
        action="store_true",
        help="Ignore cached pickles and rebuild every barangay model from scratch."
    )
    args = parser.parse_args()

    train_model(force_retrain=args.force_retrain)


if __name__ == "__main__":
    run_cli()
