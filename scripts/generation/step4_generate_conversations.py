import argparse
import json
import ollama
import operator
import os
import pyserini.search
import random
import re
import sys
import time
import traceback
import transformers
from typing import Dict, List


def generate_catalogue_text(products_list: List[str], products_data: Dict[str, dict]) -> str:
    catalogue_text = ""

    for i, x in enumerate(products_list, start=1):
        pd = products_data[x]

        catalogue_text += \
            f"- [Product {i}]:\n" \
            f"-- \"Title\" = {pd['title']}\n" \
            f"-- \"Description\" = {pd['description']}\n" \
            f"-- \"Price\" = ${pd['price']}\n"

        for j, y in enumerate(pd["reviews"], start=1):
            catalogue_text += \
                f"-- \"Review {j}\" = {y}\n"

        catalogue_text += "\n"

    return catalogue_text

def generate_corpus_text(documents_list: List[str]) -> str:
    corpus_text = ""

    for i, x in enumerate(documents_list, start=1):
        corpus_text += \
            f"- [Document {i}]:\n" \
            f"-- \"Text\" = {x}\n" \
            "\n"

    return corpus_text

def sparse_search(searcher, query:str, top_k: int) -> List[str]:
    # # Temp variables used to extract the textual content.
    # str_contents = "\"contents\" : \""
    # len_contents = len(str_contents)

    hits = [str(x.docid) for x in searcher.search(query, top_k)]

    # Extract the text from the sparse Lucene-based index.
    return [json.loads(searcher.doc(x).raw())["segment"] for x in hits]

    # raws = {x: searcher.doc(x).raw()for x in hits}
    # sis = {x: y.index(str_contents) + len_contents for x, y in raws.items()}
    # texts = {x: re.sub("\\s+", " ", y[sis[x]: -3]) for x, y in raws.items()}
    # del raws, sis
    #
    # return [texts[x] for x in hits]

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
    args.num_products = 10
    args.num_documents = 10
    args.source_filename = f"{os.environ['CAMEO_DATA']}{os.sep}AmazonReviews{os.sep}source_conversations_data.jsonl"
    args.sparse_index_folder = f"{os.environ['CAMEO_DATA']}{os.sep}MS_MARCOv2_1{os.sep}sparse_index"
    args.query_prompt_filename = f"..{os.sep}..{os.sep}prompts{os.sep}product_to_query_prompt.txt"
    args.conv_prompt_filename = f"..{os.sep}..{os.sep}prompts{os.sep}conversation_generation_prompt.txt"
    args.query_model = "llama3.1"
    args.conv_model = "llama3.1"
    args.output_filename = f"{os.environ['CAMEO_DATA']}{os.sep}AmazonReviews{os.sep}llama31_p2_p3_conversations.jsonl"

    return args


def main():
    NS_IN_S = 1000 * 1000 * 1000

    # ------------------------------------------------------------------------------------------------------------------
    # Get the command line parameters passed to the script.
    # ------------------------------------------------------------------------------------------------------------------
    args = parse_args()

    # ------------------------------------------------------------------------------------------------------------------
    # Create the output file on disk.
    # ------------------------------------------------------------------------------------------------------------------
    with open(args.output_filename, "wt", encoding="utf-8") as _:
        pass
    del _

    # ------------------------------------------------------------------------------------------------------------------
    # Read the source data for conversations from disk.
    # ------------------------------------------------------------------------------------------------------------------
    source_data = {}

    print("Starting reading source conversations data.", flush=True)

    st = time.time_ns()
    ctr = 0
    with open(args.source_filename, "rt", encoding="utf-8") as fi:
        for line in fi:
            if line == "\n":
                continue

            data = json.loads(line)
            target = data["target"]
            source_data[target] = data
            del data, target

            ctr += 1
            if (ctr % 100000) == 0:
                print(f"Processed {ctr} source conversations data in {(time.time_ns() - st) / NS_IN_S:.3f} s.", flush=True)
    del line, fi

    print(f"Processed {ctr} source conversations data in {(time.time_ns() - st) / NS_IN_S:.3f} s.\n", flush=True)
    del st, ctr

    # ------------------------------------------------------------------------------------------------------------------
    # Read the "query" and "conversation" prompts from disk.
    # ------------------------------------------------------------------------------------------------------------------
    with open(args.query_prompt_filename, "rt", encoding="utf-8") as fi:
        query_prompt_template = fi.read()
    del fi

    with open(args.conv_prompt_filename, "rt", encoding="utf-8") as fi:
        conv_prompt_template = fi.read()
    del fi

    # ------------------------------------------------------------------------------------------------------------------
    # Load the sparse index from disk.
    # ------------------------------------------------------------------------------------------------------------------
    searcher = pyserini.search.lucene.LuceneSearcher(args.sparse_index_folder)
    searcher.set_bm25(k1=1.2, b=0.75)

    # ------------------------------------------------------------------------------------------------------------------
    # Load the Llama 3.1 model through Ollama APIs.
    # ------------------------------------------------------------------------------------------------------------------
    client = ollama.Client(host="http://localhost:11434")

    tokenizer_model = f"{os.environ['HF_HOME']}{os.sep}Meta-Llama-3.1-8B{os.sep}"
    tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_model, local_files_only=True)
    del tokenizer_model

    # ------------------------------------------------------------------------------------------------------------------
    # Generate the conversations.
    # ------------------------------------------------------------------------------------------------------------------
    # random.seed(args.rng_seed)
    random.seed(time.time_ns())

    print("Starting generating conversations.", flush=True)

    source_keys = list(source_data.keys())
    random.shuffle(source_keys)

    st = time.time_ns()
    ctr = 0
    for target in source_keys:
        try:
            data = source_data[target]
            target_title = data["target_title"]
            target_text = data["target_text"]
            # catalogue_text = data["catalogue_text"]

            # ----------------------------------------------------------------------------------------------------------
            # Generate the plausible query from the target product.
            # ----------------------------------------------------------------------------------------------------------
            query_prompt = query_prompt_template.format(**{"product_text": target_text})
            del target_text
            num_tokens = len(tokenizer.encode(query_prompt, add_special_tokens=True))
            print(f"\t\tquery_prompt tokens: {num_tokens}.", flush=True)

            qt = time.time_ns()
            query = client.generate(model=args.query_model,
                                    prompt=query_prompt,
                                    options={
                                        "num_ctx": num_tokens + 100,
                                        "num_predict": 5
                                    }
                                    )["response"]
            del num_tokens
            query = re.sub("\\s+", " ", query) \
                     .replace("\"", "") \
                     .replace("?", "").lower()
            qt = time.time_ns() - qt

            print(f"\tQuery generated in {qt / NS_IN_S:.3f} s.", flush=True)
            del qt

            if query.count(" ") > 4:
                raise Exception("Invalid query.")

            # ----------------------------------------------------------------------------------------------------------
            # Retrieve some documents containing general knowledge using the generated query.
            # ----------------------------------------------------------------------------------------------------------
            kt = time.time_ns()
            corpus_data = sparse_search(searcher, query, args.num_documents)
            kt = time.time_ns() - kt

            print(f"\tRetrieval performed in {kt / NS_IN_S:.3f} s.", flush=True)
            del kt

            # ----------------------------------------------------------------------------------------------------------
            # Generate the "catalogue_text" and the "knowledge_text" from the raw data.
            # ----------------------------------------------------------------------------------------------------------
            related = data["related"][:args.num_products - 1]
            products = [target] + related
            random.shuffle(products)

            products_data = {x: y for x, y in zip(data["products"], data["products_data"])}
            products_data = {x: products_data[x] for x in products}

            catalogue_text = generate_catalogue_text(products, products_data)
            corpus_text = generate_corpus_text(corpus_data)

            # ----------------------------------------------------------------------------------------------------------
            # Generate the conversation.
            # ----------------------------------------------------------------------------------------------------------
            conv_prompt = conv_prompt_template.format(**{
                "query_text": query,
                "target_title": target_title,
                "catalogue_text": catalogue_text,
                "corpus_text": corpus_text
            })
            del target_title
            num_tokens = len(tokenizer.encode(conv_prompt, add_special_tokens=True))
            print(f"\t\tconv_prompt tokens: {num_tokens}.", flush=True)

            ct = time.time_ns()
            conversation = client.generate(model=args.conv_model,
                                           prompt=conv_prompt,
                                           options={
                                               "num_ctx": num_tokens + 1100,
                                               "num_predict": 1000
                                           }
                                           )["response"]
            del num_tokens
            conversation = re.sub("\\s+", " ", conversation)
            ct = time.time_ns() - ct

            print(f"\tConversation generated in {ct / NS_IN_S:.3f} s.", flush=True)
            del ct

            # ----------------------------------------------------------------------------------------------------------
            # Write the conversation and related data to disk.
            # ----------------------------------------------------------------------------------------------------------
            with open(args.output_filename, "at", encoding="utf-8") as fo:
                print(json.dumps({
                    "target": target,
                    "related": related,
                    "products": products,
                    "products_data": products_data,
                    "target_title": data["target_title"],
                    "target_text": data["target_text"],
                    "catalogue_text": catalogue_text,
                    "corpus_text": corpus_text,
                    "query_prompt": query_prompt,
                    "query": query,
                    "conversation_prompt": conv_prompt,
                    "conversation": conversation
                }), file=fo, flush=True)
            del fo
        except:
            print(f"** Unable to process the product {ctr} (\"{target}\").", flush=True)
            traceback.print_exc()
            sys.stdout.flush()
            sys.stderr.flush()


        ctr += 1
        print(f"Processed {ctr} conversations (target \"{target}\") in {(time.time_ns() - st) / NS_IN_S:.3f} s.", flush=True)
    del st, ctr

    print("\nDone!\n", flush=True)


if __name__ == '__main__':
    main()
