import argparse
import gzip
import json
import math
import operator
import os
import random
import statistics
import time


def parse_args() -> argparse.Namespace:
    # args = argparse.ArgumentParser()
    # args.add_argument("--output_filename",
    #                   dest="output_filename",
    #                   type=str,
    #                   required=True)
    #
    # return args.parse_args()

    args = argparse.Namespace()

    args.random_seed = 20241225

    args.catalogue_filename = f"{os.environ['CAMEO_DATA']}{os.sep}AmazonReviews{os.sep}catalogue.jsonl.gz"
    args.catalogue_size = 12312760

    args.num_products = 10000
    args.output_filename = f"{os.environ['CAMEO_DATA']}{os.sep}AmazonReviews{os.sep}selected_products.jsonl"

    return args


def main():
    NS_IN_S = 1000 * 1000 * 1000

    # ------------------------------------------------------------------------------------------------------------------
    # Get the command line parameters passed to the script.
    # ------------------------------------------------------------------------------------------------------------------
    args = parse_args()

    # ------------------------------------------------------------------------------------------------------------------
    # Extract 'num_products' products at random from the catalogue.
    # ------------------------------------------------------------------------------------------------------------------
    # Set the seed of the random number generator.
    random.seed(args.random_seed)

    # Select 'num_products' random products from the catalogue.
    selected_products_idx = sorted(random.sample(range(args.catalogue_size), args.num_products))

    st = time.time_ns()
    # Parse the catalogue to extract the products' data.
    with open(args.output_filename, "wt", encoding="utf-8") as fo, \
         gzip.open(args.catalogue_filename, "rt", encoding="utf-8") as fi:
        ctr = 0
        for line in fi:
            if ctr == selected_products_idx[0]:
                # Save the extracted product data to disk.
                print(line, end="", file=fo, flush=True)

                # Skip to the next selected product.
                selected_products_idx = selected_products_idx[1:]
                if len(selected_products_idx) == 0:
                    break

            ctr += 1
            if (ctr % 100000) == 0:
                print(f"Processed {ctr} products in {(time.time_ns() - st) / NS_IN_S:.3f} s.", flush=True)

    print(f"Processed {ctr} products in {(time.time_ns() - st) / NS_IN_S:.3f} s.", flush=True)
    print("\nDone!\n", flush=True)


if __name__ == '__main__':
    main()
