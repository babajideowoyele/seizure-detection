"""Run this locally to train all models and save checkpoints."""
from config import Config
from train import train_all_models

if __name__ == "__main__":
    config = Config()
    train_all_models(config)
