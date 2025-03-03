import argparse
import json
import ollama
import operator
import os
import pyserini.search.faiss
import pyserini.search.lucene
import random
import re
import sys
import time
import traceback
import transformers
from typing import Dict, List, Optional, Tuple


class Retriever:
    def __init__(self, sparse_index_folder: str, dense_index_folder: Optional[str], k1: float = 1.2, b: float = 0.75):
        # Check that the first parameter must be provided.
        assert sparse_index_folder is not None
        # Check that the sparse index parameter is the path to a valid directory.
        assert isinstance(sparse_index_folder, str) and os.path.exists(sparse_index_folder) and \
               os.path.isdir(sparse_index_folder)
        # Check that the dense index parameter is the path to a valid directory.
        assert dense_index_folder is None or (isinstance(dense_index_folder, str) and
                                              os.path.exists(dense_index_folder) and os.path.isdir(dense_index_folder))

        # Load the sparse index from disk.
        self.sparse_searcher = pyserini.search.lucene.LuceneSearcher(sparse_index_folder)
        self.sparse_searcher.set_bm25(k1=k1, b=b)
        del sparse_index_folder

        if dense_index_folder is not None:
            # Load the dense index from disk.
            self.dense_searcher = pyserini.search.faiss.FaissSearcher(dense_index_folder,
                                                                      "castorini/tct_colbert-v2-hnp-msmarco")
        else:
            # Set the dense searcher as None, since it is not provided.
            self.dense_searcher = None
        del dense_index_folder


    def __call__(self, query: str, top_k: int) -> List[Tuple[str, str]]:
        assert query is not None and isinstance(query, str)
        assert top_k is not None and isinstance(top_k, int) and top_k > 0

        text_keys = { "contents", "segment" }

        # Perform the search using the sparse index.
        sparse_hits = {str(x.docid): i for i, x in enumerate(self.sparse_searcher.search(query, top_k), start=1)}

        if self.dense_searcher is not None:
            # Perform the search using the sparse index.
            dense_hits = {str(x.docid): i for i, x in enumerate(self.dense_searcher.search(query, top_k), start=1)}
            del query

            # Determine the stratus for each document.
            stratum_hits = {k: (sparse_hits.get(k, top_k + 1), dense_hits.get(k, top_k + 1))
                            for k in sparse_hits.keys() | dense_hits.keys()}
            stratum_hits = {k: (min(v[0], v[1]), max(v[0], v[1])) for k, v in stratum_hits.items()}
            del sparse_hits, dense_hits
        else:
            stratum_hits = {k: (v, top_k + 1) for k, v in sparse_hits.items()}
            del sparse_hits, query

        # Select the 'top k' results from stratum data, then sort by stratum in ascending order.
        stratum_hits = [(v[0], v[1], k) for k, v in stratum_hits.items()]
        stratum_hits = sorted(stratum_hits, key=operator.itemgetter(0, 1), reverse=False)[:top_k]

        # Extract the text from the sparse Lucene-based index.
        raws = {k: json.loads(self.sparse_searcher.doc(k).raw()) for _, _, k in stratum_hits}
        texts = {k: v[next(x for x in text_keys & v.keys())] for k, v in raws.items()}
        del text_keys, raws

        return [(k, texts[k]) for _, _, k, in stratum_hits]


def parse_args() -> argparse.Namespace:
    args = argparse.ArgumentParser()
    args.add_argument("--random_seed",
                      dest="random_seed",
                      type=int,
                      default=20250218,  # Submission deadline for Resource Paper @SIGIR 2025
                      required=False)

    args.add_argument("--bm25_k1",
                      dest="bm25_k1",
                      type=float,
                      default=1.2,
                      required=False)
    args.add_argument("--bm25_b",
                      dest="bm25_b",
                      type=float,
                      default=0.75,
                      required=False)

    args.add_argument("--catalogue_filename",
                      dest="catalogue_filename",
                      type=str,
                      required=True)
    args.add_argument("--catalogue_size",
                      dest="catalogue_size",
                      type=int,
                      default=12312760,
                      required=False)
    args.add_argument("--num_random_products",
                      dest="num_random_products",
                      type=int,
                      required=True)
    args.add_argument("--num_reviews",
                      dest="num_reviews",
                      type=int,
                      default=3,
                      required=False)

    args.add_argument("--catalogue_sparse_index_folder",
                      dest="catalogue_sparse_index_folder",
                      type=str,
                      required=True)
    args.add_argument("--catalogue_dense_index_folder",
                      dest="catalogue_dense_index_folder",
                      type=str,
                      required=True)
    args.add_argument("--num_products",
                      dest="num_products",
                      type=int,
                      default=10,
                      required=False)
    args.add_argument("--documents_sparse_index_folder",
                      dest="documents_sparse_index_folder",
                      type=str,
                      required=True)
    # args.add_argument("--documents_dense_index_folder",
    #                   dest="documents_dense_index_folder",
    #                   type=str,
    #                   required=True)
    args.add_argument("--num_documents",
                      dest="num_documents",
                      type=int,
                      default=10,
                      required=False)

    args.add_argument("--query_prompt_filename",
                      dest="query_prompt_filename",
                      type=str,
                      required=True)
    args.add_argument("--query_tokenizer",
                      dest="query_tokenizer",
                      type=str,
                      default="meta-llama/Llama-3.1-8B",
                      required=False)
    args.add_argument("--query_model",
                      dest="query_model",
                      type=str,
                      default="llama3.1",
                      required=False)
    args.add_argument("--query_max_words",
                      dest="query_max_words",
                      type=int,
                      default=7,
                      required=False)
    args.add_argument("--conv_prompt_filename",
                      dest="conv_prompt_filename",
                      type=str,
                      required=True)
    args.add_argument("--conv_tokenizer",
                      dest="conv_tokenizer",
                      type=str,
                      default="meta-llama/Llama-3.1-8B",
                      required=False)
    args.add_argument("--conv_model",
                      dest="conv_model",
                      type=str,
                      default="llama3.1",
                      required=False)

    args.add_argument("--create_output_file",
                      dest="create_output_file",
                      type=bool,
                      default=True,
                      required=False)
    args.add_argument("--output_filename",
                      dest="output_filename",
                      type=str,
                      required=True)

    return args.parse_args()


def generate_catalogue_text(products_list: List[str], products_data: Dict[str, dict], num_reviews: int = 3) -> str:
    catalogue_text = ""

    for i, x in enumerate(products_list, start=1):
        pd = products_data[x]

        catalogue_text += \
            f"- [Product {i}]:\n" \
            f"-- \"Title\" = {pd['title']}\n" \
            f"-- \"Description\" = {pd['description']}\n" \
            f"-- \"Price\" = ${pd['price']}\n"

        reviews = [(x["helpful"], x["timestamp"], f"{x['title']} {x['text']}") for x in pd["valid_reviews"]]
        reviews = sorted(reviews, key=operator.itemgetter(0, 1), reverse=True)[:num_reviews]
        for j, y in enumerate(reviews, start=1):
            catalogue_text += \
                f"-- \"Review {j}\" = {y[2]}\n"

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


def generate_text(prompt: str, model: str, num_ctx: int, num_predict: int, client, tokenizer) -> str:
    assert prompt is not None
    assert isinstance(prompt, str)

    assert model is not None
    assert isinstance(model, str)

    assert num_ctx is not None
    assert isinstance(num_ctx, int)
    assert num_ctx > 0

    assert num_predict is not None
    assert isinstance(num_predict, int)
    assert num_predict > 0

    assert client is not None

    assert tokenizer is not None

    num_tokens = len(tokenizer.encode(prompt, add_special_tokens=True))
    result = client.generate(model=model, prompt=prompt, options={
                                "num_ctx": num_tokens + num_ctx,
                                "num_predict": num_predict
                            })["response"]
    del num_tokens

    return result


def main():
    NS_IN_S = 1000 * 1000 * 1000

    # ------------------------------------------------------------------------------------------------------------------
    # Get the command line parameters passed to the script.
    # ------------------------------------------------------------------------------------------------------------------
    args = parse_args()
    random.seed(args.random_seed)

    # ------------------------------------------------------------------------------------------------------------------
    # Read the "query" prompt from disk.
    # ------------------------------------------------------------------------------------------------------------------
    with open(args.query_prompt_filename, "rt", encoding="utf-8") as fi:
        query_prompt_template = fi.read()
    del fi

    with open(args.conv_prompt_filename, "rt", encoding="utf-8") as fi:
        conv_prompt_template = fi.read()
    del fi

    # ------------------------------------------------------------------------------------------------------------------
    # Extract 'num_random_products' products at random from the catalogue.
    # ------------------------------------------------------------------------------------------------------------------
    products_idx = sorted(random.sample(range(args.catalogue_size), args.num_random_products))
    products_data = {}

    st = time.time_ns()
    ctr = 0
    with open(args.catalogue_filename, "rt", encoding="utf-8") as fi:
        for line in fi:
            if line == "\n":
                continue

            if ctr == products_idx[0]:
                data = json.loads(line)
                asin = data["parent_asin"]
                products_data[asin] = data
                del data, asin

                # Skip to the next selected product.
                products_idx = products_idx[1:]
                if len(products_idx) == 0:
                    break

            ctr += 1
            if (ctr % 100000) == 0:
                print(f"Processed {ctr} products in {(time.time_ns() - st) / NS_IN_S:.3f} s.", flush=True)
        del line
    del fi

    print(f"Processed {args.catalogue_size} products in {(time.time_ns() - st) / NS_IN_S:.3f} s.", flush=True)
    print("", flush=True)
    del products_idx, st, ctr

    # ------------------------------------------------------------------------------------------------------------------
    # Load the products retriever using the sparse and dense indexes stored on disk.
    # ------------------------------------------------------------------------------------------------------------------
    st = time.time_ns()
    retr = Retriever(args.catalogue_sparse_index_folder, args.catalogue_dense_index_folder, args.bm25_k1, args.bm25_b)
    print(f"Products retriever loaded successfully in {(time.time_ns() - st) / NS_IN_S:.3f} s.\n", flush=True)
    print("", flush=True)
    del st

    # ------------------------------------------------------------------------------------------------------------------
    # Perform retrieval of 'num_products' most similar products based on the target product.
    # ------------------------------------------------------------------------------------------------------------------
    products_related_idx = {}

    st = time.time_ns()
    ctr = 0
    for asin, data in products_data.items():
        title = data["title"]
        description = "\n".join(data["description"])
        text = f"{title}\n{description}"
        del title, description

        # Retrieve the most similar products to the current one, and remove it from the results, if present.
        products_related_idx[asin] = [k for k, _ in retr(text, args.num_products)
                                      if k != asin][:args.num_products - 1]
        del text

        ctr += 1
        print(f"Retrieved related products for {ctr} products in {(time.time_ns() - st) / NS_IN_S:.3f} s.", flush=True)
    del asin, data

    print("", flush=True)
    del st, ctr

    # ------------------------------------------------------------------------------------------------------------------
    # Deallocate the products retriever object.
    # ------------------------------------------------------------------------------------------------------------------
    del retr

    # ------------------------------------------------------------------------------------------------------------------
    # Extract the data about related products from the catalogue.
    # ------------------------------------------------------------------------------------------------------------------
    products_idx = {k2 for k1, v1 in products_related_idx.items() for k2 in v1}
    products_idx = set(products_idx | products_data.keys())

    st = time.time_ns()
    ctr = 0
    with open(args.catalogue_filename, "rt", encoding="utf-8") as fi:
        for line in fi:
            if line == "\n":
                continue

            data = json.loads(line)
            asin = data["parent_asin"]

            if asin in products_idx:
                products_data[asin] = data

            del data, asin

            ctr += 1
            if (ctr % 100000) == 0:
                print(f"Processed {ctr} products in {(time.time_ns() - st) / NS_IN_S:.3f} s.", flush=True)
        del line
    del fi

    print(f"Processed {args.catalogue_size} products in {(time.time_ns() - st) / NS_IN_S:.3f} s.", flush=True)
    print("", flush=True)
    del products_idx, st, ctr

    # ------------------------------------------------------------------------------------------------------------------
    # Load the documents retriever using the sparse and dense indexes stored on disk.
    # ------------------------------------------------------------------------------------------------------------------
    st = time.time_ns()
    retr = Retriever(args.documents_sparse_index_folder, None, args.bm25_k1, args.bm25_b)
    # retr = Retriever(args.documents_sparse_index_folder, args.documents_dense_index_folder, args.bm25_k1, args.bm25_b)
    print(f"Documents retriever loaded successfully in {(time.time_ns() - st) / NS_IN_S:.3f} s.\n", flush=True)
    print("", flush=True)
    del st

    # ------------------------------------------------------------------------------------------------------------------
    # Load the Llama 3.1 model through Ollama APIs.
    # ------------------------------------------------------------------------------------------------------------------
    client = ollama.Client(host="http://localhost:11434")

    q_tokenizer = transformers.AutoTokenizer.from_pretrained(args.query_tokenizer)
    if args.conv_tokenizer == args.query_tokenizer:
        c_tokenizer = q_tokenizer
    else:
        c_tokenizer = transformers.AutoTokenizer.from_pretrained(args.conv_tokenizer)

    # ------------------------------------------------------------------------------------------------------------------
    # Create the output file on disk.
    # ------------------------------------------------------------------------------------------------------------------
    if args.create_output_file is not None and isinstance(args.create_output_file, bool) and args.create_output_file:
        with open(args.output_filename, "wt", encoding="utf-8") as _:
            pass
        del _

    # ------------------------------------------------------------------------------------------------------------------
    # Retrieve related documents, using a LLM-generated query from the target product data.
    # Then, generate the conversation and save it to disk.
    # ------------------------------------------------------------------------------------------------------------------
    st = time.time_ns()
    ctr = 0
    for target_asin, related_asins in products_related_idx.items():
        try:
            # Extract product metadata about target and related products.
            target_data = products_data[target_asin]
            related_data = {k: products_data[k] for k in related_asins}
            catalogue_data = related_data | {target_asin: target_data}

            # Randomly shuffle the products for the final list in the generation prompt.
            products_idx = [target_asin] + related_asins
            random.shuffle(products_idx)

            # ----------------------------------------------------------------------------------------------------------
            # Generate the plausible query from the target product.
            # ----------------------------------------------------------------------------------------------------------
            target_text = \
                f"\"Title\" = {target_data['title']}\n" \
                f"\"Description\" = {' '.join(target_data['description'])}\n"
            query_prompt = query_prompt_template.format(**{"product_text": target_text})

            qt = time.time_ns()
            query = generate_text(query_prompt, args.query_model, 100, 100, client, q_tokenizer)
            query = re.sub("\\s+", " ", query) \
                    .replace("\"", "") \
                    .replace("?", "") \
                    .strip().lower()
            qt = time.time_ns() - qt

            if query.count(" ") >= args.query_max_words:
                raise Exception("Invalid query: too much words.")

            print(f"\tQuery generated in {qt / NS_IN_S:.3f} s.", flush=True)
            del qt

            # ----------------------------------------------------------------------------------------------------------
            # Retrieve documents containing general knowledge using the generated query.
            # ----------------------------------------------------------------------------------------------------------
            rt = time.time_ns()
            corpus_data = retr(query, args.num_documents)
            corpus_data = [re.sub("\\s+", " ", x).strip() for _, x in corpus_data]
            rt = time.time_ns() - rt

            print(f"\tRetrieval performed in {rt / NS_IN_S:.3f} s.", flush=True)
            del rt

            # ----------------------------------------------------------------------------------------------------------
            # Generate the "catalogue_text" and the "knowledge_text" from the raw data.
            # ----------------------------------------------------------------------------------------------------------
            catalogue_text = generate_catalogue_text(products_idx, catalogue_data, args.num_reviews)
            corpus_text = generate_corpus_text(corpus_data)

            # ----------------------------------------------------------------------------------------------------------
            # Generate the conversation.
            # ----------------------------------------------------------------------------------------------------------
            conv_prompt = conv_prompt_template.format(**{
                "query_text": query,
                "target_title": target_data["title"],
                "catalogue_text": catalogue_text,
                "corpus_text": corpus_text
            })

            ct = time.time_ns()
            conversation = generate_text(conv_prompt, args.conv_model, 1100, 1000, client,
                                         c_tokenizer)
            conversation = re.sub("\\s+", " ", conversation)\
                .replace(" U: ", "\nU: ").replace(" S: ", "\nS: ").strip()
            ct = time.time_ns() - ct

            print(f"\tConversation generated in {ct / NS_IN_S:.3f} s.", flush=True)
            del ct

            # ----------------------------------------------------------------------------------------------------------
            # Write the conversation and related data to disk.
            # ----------------------------------------------------------------------------------------------------------
            with open(args.output_filename, "at", encoding="utf-8") as fo:
                print(json.dumps({
                    "target": target_asin,
                    "related": related_asins,
                    "products": products_idx,
                    "products_data": catalogue_data,
                    "target_title": target_data["title"],
                    "target_text": target_text,
                    "catalogue_text": catalogue_text,
                    "corpus_text": corpus_text,
                    "query_prompt": query_prompt,
                    "query": query,
                    "conversation_prompt": conv_prompt,
                    "conversation": conversation
                }), file=fo, flush=True)
            del fo

            del target_data, related_data, products_idx, target_text, query_prompt, query, conv_prompt, conversation
            del catalogue_data, corpus_data
        except:
            print(f"** Unable to process the product {ctr} (\"{target_asin}\").", flush=True)
            traceback.print_exc()
            sys.stdout.flush()
            sys.stderr.flush()

        ctr += 1
        print(f"Processed {ctr} conversations (target \"{target_asin}\") in {(time.time_ns() - st) / NS_IN_S:.3f} s.",
              flush=True)
    del st, ctr

    # ------------------------------------------------------------------------------------------------------------------
    # Deallocate the documents retriever, Ollama client, tokenizer, and other objects.
    # ------------------------------------------------------------------------------------------------------------------
    del products_related_idx, products_data, client, retr, q_tokenizer
    try:
        del c_tokenizer
    except:
        pass

    print("\nDone!\n", flush=True)


if __name__ == '__main__':
    main()

