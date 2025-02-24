import argparse
import json
import operator
import os
import retrieval
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
    args.sparse_index_folder = f"{os.environ['CAMEO_DATA']}{os.sep}AmazonReviews{os.sep}sparse_index"
    args.dense_index_folder = f"{os.environ['CAMEO_DATA']}{os.sep}AmazonReviews{os.sep}dense_index"
    args.products_filename = f"{os.environ['CAMEO_DATA']}{os.sep}AmazonReviews{os.sep}selected_products.jsonl"
    args.num_products = 20
    args.output_filename = f"{os.environ['CAMEO_DATA']}{os.sep}AmazonReviews{os.sep}data_for_conversations.jsonl"

    return args


def main():
    NS_IN_S = 1000 * 1000 * 1000

    print("Starting execution.", flush=True)

    # ------------------------------------------------------------------------------------------------------------------
    # Get the command line parameters passed to the script.
    # ------------------------------------------------------------------------------------------------------------------
    args = parse_args()

    # ------------------------------------------------------------------------------------------------------------------
    # Create the output file on disk.
    # ------------------------------------------------------------------------------------------------------------------
    with open(args.output_filename, "wt", encoding="utf-8") as fo:
        pass
    del fo

    # ------------------------------------------------------------------------------------------------------------------
    # Load the selected products data from disk.
    # ------------------------------------------------------------------------------------------------------------------
    products = {}

    st = time.time_ns()
    ctr = 0
    with open(args.products_filename, "rt", encoding="utf-8") as fi:
        for line in fi:
            if line == "\n":
                continue

            data = json.loads(line)
            asin = data["parent_asin"]
            title = data["title"]
            description = "\n".join(data["description"])
            text = f"{title}\n{description}"
            del data, title, description

            products[asin] = text
            del asin, text

            ctr += 1
            if (ctr % 100000) == 0:
                print(f"Processed {ctr} products in {(time.time_ns() - st) / NS_IN_S:.3f} s.", flush=True)

    print(f"Processed {ctr} products in {(time.time_ns() - st) / NS_IN_S:.3f} s.\n", flush=True)
    del st, ctr
    del line, fi

    # ------------------------------------------------------------------------------------------------------------------
    # Load the retriever using the sparse and dense indexes stored on disk.
    # ------------------------------------------------------------------------------------------------------------------
    st = time.time_ns()
    retr = retrieval.Retriever(args.sparse_index_folder, args.dense_index_folder)
    print(f"Retriever created successfully in {(time.time_ns() - st) / NS_IN_S:.3f} s.\n", flush=True)
    del st

    # ------------------------------------------------------------------------------------------------------------------
    # Find the top-k most similar products to each of the selected products.
    # ------------------------------------------------------------------------------------------------------------------
    data_conversations = {}
    print("Starting retrieval.", flush=True)

    st = time.time_ns()
    ctr = 0
    for asin, text in sorted(products.items(), key=operator.itemgetter(0), reverse=False):
        try:
            # Retrieve the most similar products to the current one, and remove it from the results, if present.
            result = retr.search(text, args.num_products + 1)
            result.pop(asin, None)

            data_conversations[asin] = set(result.keys())
            del result

            with open(args.output_filename, "at", encoding="utf-8") as fo:
                print(json.dumps({
                    "target": asin,
                    "related": sorted(data_conversations[asin])
                }), file=fo, flush=True)
            del fo
        except:
            print(f"** Unable to process the product {ctr} (\"{asin}\").")

        ctr += 1
        print(f"Product {ctr} (\"{asin}\") processed in {(time.time_ns() - st) / NS_IN_S:.3f} s.", flush=True)

    print(f"Processed {ctr} products in {(time.time_ns() - st) / NS_IN_S:.3f} s.\n", flush=True)
    del st, ctr
    del asin, text, products

    # ------------------------------------------------------------------------------------------------------------------
    # Deallocate the retriever object.
    # ------------------------------------------------------------------------------------------------------------------
    del retr

    # # ------------------------------------------------------------------------------------------------------------------
    # # Write the data needed for generating conversations to the output file.
    # # ------------------------------------------------------------------------------------------------------------------
    # with open(args.output_filename, "wt", encoding="utf-8") as fo:
    #     for target, related_products in sorted(data_conversations.items(), key=operator.itemgetter(0), reverse=False):
    #         print(json.dumps({
    #             "target": target,
    #             "related": sorted(related_products)
    #         }), file=fo)
    #
    #     fo.flush()
    # del fo

    print("\nDone!\n", flush=True)


if __name__ == '__main__':
    main()
