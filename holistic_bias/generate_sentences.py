#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import argparse

from holistic_bias.src.sentences import HolisticBiasSentenceGenerator
from holistic_bias.src.util import DEFAULT_DATASET_VERSION


if __name__ == "__main__":

# Path to base dataset folder
    parser = argparse.ArgumentParser("Generate HolisticBias sentences")
    parser.add_argument(
        "--dataset-version",
        type=str,
        default=DEFAULT_DATASET_VERSION,
        help="Which version of the dataset to load",
    )
    parser.add_argument(
        "--use-small-set",
        action="store_true",
        help="Include only a small subset of the total number of descriptors, for tractability",
    )
    args = parser.parse_args()

    save_folder = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "dataset", args.dataset_version
    )
    print(
        f"Instantiating the HolisticBias sentence generator and saving output files to {save_folder}."
    )
    generator = HolisticBiasSentenceGenerator(
        save_folder=save_folder,
        dataset_version=args.dataset_version,
        use_small_set=args.use_small_set,
    )
    print(f"\nSample sentences:")
    for _ in range(5):
        sentence_with_metadata = generator.get_sentence()
        print("\t" + sentence_with_metadata["text"])
