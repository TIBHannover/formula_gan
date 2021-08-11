import sys
import os
import re

import argparse
import logging

import torch
from pytorch_lightning import Trainer
import wap_model
import formula_model
from config import Config


def main():

    config = Config()

    print(config.to_args())
    print(vars(config.to_flat_args()))

    model = formula_model.FormulaOCR(config.to_args(), flat_params=config.to_flat_args())

    # most basic trainer, uses good defaults
    trainer = Trainer()
    trainer.fit(model)


if __name__ == "__main__":
    sys.exit(main())
