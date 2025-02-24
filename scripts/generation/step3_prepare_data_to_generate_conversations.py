import argparse
import json
import ollama
import operator
import os
import random
import re
import sys
import time
import traceback
import transformers


def parse_args() -> argparse.Namespace:
    # args = argparse.ArgumentParser()
    # args.add_argument("--output_filename",
    #                   dest="output_filename",
    #                   type=str,
    #                   required=True)
    #
    # return args.parse_args()

    args = argparse.Namespace()
    args.rng_seed = 20241225
    args.max_num_reviews = 3
    args.source_filename = f"{os.environ['CAMEO_DATA']}{os.sep}AmazonReviews{os.sep}data_for_conversations.jsonl"
    args.catalogue_filename = f"{os.environ['CAMEO_DATA']}{os.sep}AmazonReviews{os.sep}catalogue.jsonl"
    args.output_filename = f"{os.environ['CAMEO_DATA']}{os.sep}AmazonReviews{os.sep}source_conversations_data.jsonl"

    return args


def main():
    NS_IN_S = 1000 * 1000 * 1000

    # ------------------------------------------------------------------------------------------------------------------
    # Get the command line parameters passed to the script.
    # ------------------------------------------------------------------------------------------------------------------
    args = parse_args()

    # ------------------------------------------------------------------------------------------------------------------
    # Read the source data for conversations from disk.
    # ------------------------------------------------------------------------------------------------------------------
    data_conversations = {}
    products = set()

    print("Starting reading target-related data.", flush=True)

    st = time.time_ns()
    ctr = 0
    with open(args.source_filename, "rt", encoding="utf-8") as fi:
        for line in fi:
            if line == "\n":
                continue

            data = json.loads(line)
            target = data["target"]
            related = data["related"]
            del data

            data_conversations[target] = related
            products.add(target)
            products.update(related)
            del target, related

            ctr += 1
            if (ctr % 100000) == 0:
                print(f"Processed {ctr} target-related in {(time.time_ns() - st) / NS_IN_S:.3f} s.", flush=True)
    del line, fi

    print(f"Processed {ctr} target-related in {(time.time_ns() - st) / NS_IN_S:.3f} s.\n", flush=True)
    del st, ctr

    # ------------------------------------------------------------------------------------------------------------------
    # Extract the products information from the catalogue.
    # ------------------------------------------------------------------------------------------------------------------
    data_products = {}

    print("Starting reading product data.", flush=True)

    st = time.time_ns()
    ctr = 0
    with open(args.catalogue_filename, "rt", encoding="utf-8") as fi:
        for line in fi:
            if line == "\n":
                continue

            ctr += 1
            if (ctr % 100000) == 0:
                print(f"Processed {ctr} products in {(time.time_ns() - st) / NS_IN_S:.3f} s.", flush=True)

            data = json.loads(line)
            asin = data["parent_asin"]
            if asin not in products:
                del data, asin
                continue

            title = re.sub("\\s+", " ", data["title"]).replace("<br />", "")
            description = " ".join(re.sub("\\s+", " ", x).replace("<br />", "")
                                   for x in data["description"])
            price = data["price"]
            reviews = sorted([{
                "title": re.sub("\\s+", " ", x["title"]).replace("<br />", ""),
                "text": re.sub("\\s+", " ", x["text"]).replace("<br />", ""),
                "rating": x["rating"],
                "helpful": x["helpful"]
            } for x in data["reviews"] if x["verified purchase"] is True and x["helpful"] > 0],
                key=operator.itemgetter("helpful"), reverse=True)[:args.max_num_reviews]
            del data

            data_products[asin] = {
                "title": title,
                "description": description,
                "price": price,
                "reviews": [f"{x['title']}: {x['text']}" for x in reviews]
            }
            del title, description, price, reviews

            # Exit to process the catalogue if all products data have been extracted.
            if len(data_products) == len(products):
                break
    del line, fi

    print(f"Processed {ctr} products in {(time.time_ns() - st) / NS_IN_S:.3f} s.\n", flush=True)
    del st, ctr
    del products

    # ------------------------------------------------------------------------------------------------------------------
    # Join the target-related data with the information coming from the catalogue.
    # ------------------------------------------------------------------------------------------------------------------
    print("Starting joining data.", flush=True)

    st = time.time_ns()
    ctr = 0

    with open(args.output_filename, "wt", encoding="utf-8") as fo:
        for target, related in sorted(data_conversations.items()):
            try:
                products = [target] + related
                random.shuffle(products)

                products_data = [data_products[x] for x in products]

                catalogue_text = \
                    "[Catalogue]:\n" \
                    "\n"
                for i, x in enumerate(products, start=1):
                    pd = data_products[x]
                    catalogue_text += \
                        f"\t[Product {i}]:\n" \
                        f"\t\t\"Title\" = {pd['title']}\n" \
                        f"\t\t\"Description\" = {pd['description']}\n" \
                        f"\t\t\"Price\" = ${pd['price']}\n"
                    for j, y in enumerate(pd["reviews"], start=1):
                        catalogue_text += \
                            f"\t\t[Review {j}] = {y}\n"
                    catalogue_text += "\n"

                target_data = data_products[target]

                target_title = target_data["title"]
                target_text = \
                    f"\"Title\" = {target_data['title']}\n" \
                    f"\"Description\" = {target_data['description']}\n"
                for i, text in enumerate(target_data["reviews"], start=1):
                    target_text += f"\"Review {i}\" = {text}\n"

                del target_data

                print(json.dumps({
                    "target": target,
                    "related": related,
                    "products": products,
                    "products_data": products_data,
                    "target_title": target_title,
                    "target_text": target_text,
                    "catalogue_text": catalogue_text
                }), file=fo, flush=True)
            except:
                print(f"** Unable to process the product {ctr} (\"{target}\").", flush=True)
                traceback.print_exc()
                sys.stdout.flush()
                sys.stderr.flush()


            ctr += 1
            print(f"Joined {ctr} target-related data (target \"{target}\") in {(time.time_ns() - st) / NS_IN_S:.3f} s.", flush=True)
    del st, ctr

    print("\nDone!\n", flush=True)


if __name__ == '__main__':
    main()
