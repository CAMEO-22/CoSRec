import argparse
import gzip
import json
import math
import operator
import os
import re


def parse_args() -> argparse.Namespace:
    args = argparse.ArgumentParser()
    args.add_argument("--threshold_too_short_description",
                      dest="threshold_too_short_description",
                      type=int,
                      default=10)
    args.add_argument("--threshold_english_title_ascii",
                      dest="threshold_english_title_ascii",
                      type=float,
                      default=0.5)
    args.add_argument("--threshold_english_description_ascii",
                      dest="threshold_english_description_ascii",
                      type=float,
                      default=0.8)
    args.add_argument("--metas_folder",
                      dest="metas_folder",
                      type=str,
                      required=True)
    args.add_argument("--reviews_folder",
                      dest="reviews_folder",
                      type=str,
                      required=True)
    args.add_argument("--create_output_file",
                      dest="create_output_file",
                      type=bool,
                      required=True)
    args.add_argument("--output_filename",
                      dest="output_filename",
                      type=str,
                      required=True)

    return args.parse_args()


def main():
    # ------------------------------------------------------------------------------------------------------------------
    # Get the command line parameters passed to the script.
    # ------------------------------------------------------------------------------------------------------------------
    args = parse_args()

    # ------------------------------------------------------------------------------------------------------------------
    # Determine all metas and reviews files available on disk.
    # ------------------------------------------------------------------------------------------------------------------
    metas = [x[5: -9] for x in os.listdir(args.metas_folder) if x.startswith("meta_") and x.endswith(".jsonl.gz")]
    reviews = [x[:-9] for x in os.listdir(args.reviews_folder) if x.endswith(".jsonl.gz")]
    common = set(metas) & set(reviews)

    # ------------------------------------------------------------------------------------------------------------------
    # Create the output file on disk.
    # ------------------------------------------------------------------------------------------------------------------
    if args.create_output_file is not None and isinstance(args.create_output_file, bool) and args.create_output_file:
        with open(args.output_filename, "wt", encoding="utf-8") as _:
            pass
        del _

    # ------------------------------------------------------------------------------------------------------------------
    # Define the counters to be used later in the code, and initialize them to 0.
    # ------------------------------------------------------------------------------------------------------------------
    ctr_valid_products = 0

    ctr_p_skip_no_asin = 0
    ctr_p_skip_no_title = 0
    ctr_p_skip_no_description = 0
    ctr_p_skip_no_details = 0
    ctr_p_skip_no_categories = 0
    ctr_p_skip_no_reviews = 0
    ctr_p_skip_no_store = 0
    ctr_p_skip_no_price = 0
    ctr_p_skip_no_english = 0

    ctr_r_skip_no_asin = 0
    ctr_r_skip_no_product = 0
    ctr_r_skip_no_title = 0
    ctr_r_skip_no_text = 0
    ctr_r_skip_no_timestamp = 0
    ctr_r_skip_no_user = 0
    ctr_r_skip_no_verified = 0
    ctr_r_skip_no_rating = 0
    ctr_r_skip_no_helpful = 0
    ctr_r_skip_no_english = 0

    ctr_bought_together_yes = 0
    ctr_bought_together_no = 0

    # ------------------------------------------------------------------------------------------------------------------
    # Process the entire dataset.
    # ------------------------------------------------------------------------------------------------------------------
    ctr = 0
    for cmr in sorted(common):
        re_ws = re.compile("\\s+")

        # --------------------------------------------------------------------------------------------------------------
        # Process the "meta" file currently being considered.
        # --------------------------------------------------------------------------------------------------------------
        catalogue = {}

        with gzip.open(f"{args.metas_folder}{os.sep}meta_{cmr}.jsonl.gz", "rt", encoding="utf-8") as fi:
            for line in fi:
                # ------------------------------------------------------------------------------------------------------
                # Extract the data from the jsonl record.
                # ------------------------------------------------------------------------------------------------------
                data = json.loads(line)
                parent_asin = data["parent_asin"]
                title = data["title"]
                categories = data["categories"]
                description = data["description"]
                bullet_point_features = data["features"]
                details = data["details"]
                store = data["store"]
                price = data["price"]
                avg_rating = data["average_rating"]
                num_rating = data["rating_number"]
                bought_together = data["bought_together"]
                del data

                if bought_together is None or len(bought_together) <= 0:
                    ctr_bought_together_no += 1
                else:
                    ctr_bought_together_yes += 1

                # ------------------------------------------------------------------------------------------------------
                # Check if the product record should be discarded.
                # ------------------------------------------------------------------------------------------------------
                # 1) Discard the product if it has an invalid asin.
                if parent_asin is None or (not isinstance(parent_asin, str)) or len(parent_asin) <= 0:
                    ctr_p_skip_no_asin += 1
                    continue
                # 2) Discard the product if it doesn't have name of it is empty.
                if title is None or (not isinstance(title, str)) or len(title) <= 0:
                    ctr_p_skip_no_title += 1
                    continue
                # 3) Discard the product if it doesn't have a description of it is empty.
                if (description is None or (not isinstance(description, list)) or
                    (not all(isinstance(x, str) for x in description))) or \
                   (bullet_point_features is None or (not isinstance(bullet_point_features, list)) or
                    (not all(isinstance(x, str) for x in bullet_point_features))):
                    ctr_p_skip_no_description += 1
                    continue
                if len(description) <= 0 and len(bullet_point_features) <= 0:
                    ctr_p_skip_no_description += 1
                    continue
                # 4) Discard the product if it doesn't have details of it is empty.
                if details is None or (not isinstance(details, dict)) or len(details) <= 0 or \
                        (not all(isinstance(x, str) for x in details.keys())):
                    ctr_p_skip_no_details += 1
                    continue
                # 5) Discard the product if it doesn't have categories of it is an empty list.
                if categories is None or (not isinstance(categories, list)) or len(categories) <= 0 or \
                        (not all(isinstance(x, str) for x in categories)) or \
                        (len(categories) == 1 and len(categories[0]) <= 0):
                    ctr_p_skip_no_categories += 1
                    continue
                # 6) No reviews can be found for this product.
                if (num_rating is None or (not isinstance(num_rating, int)) or num_rating <= 0) or \
                        (avg_rating is None or (not isinstance(avg_rating, float)) or
                         (not math.isfinite(avg_rating)) or avg_rating < 0.0):
                    ctr_p_skip_no_reviews += 1
                    continue
                # 7) Discard the product if it has no store.
                if store is None or (not isinstance(store, str)) or len(store) <= 0:
                    ctr_p_skip_no_store += 1
                    continue
                # 8) Discard the product if it has no price.
                if price is None or (not isinstance(price, (float, str))):
                    ctr_p_skip_no_price += 1
                    continue
                elif isinstance(price, str):
                    try:
                        price = float(price)
                    except ValueError:
                        ctr_p_skip_no_price += 1
                        continue
                if (not math.isfinite(price)) or price <= 0.0:
                    ctr_p_skip_no_price += 1
                    continue

                # Compute the full textual description of the product.
                full_description = " ".join(description) + " " + " ".join(bullet_point_features)
                perc_title = sum(1 for ch in title if ord(ch) <= 255) / len(title)
                perc_description = sum(1 for ch in full_description if ord(ch) <= 255) / len(full_description)

                # 9) Discard the product if the description is too short.
                if len(full_description) < args.threshold_too_short_description:
                    ctr_p_skip_no_description += 1
                    continue
                # 10) Discard the product if the name of the description is likely not in english language.
                if perc_title < args.threshold_english_title_ascii or \
                        perc_description < args.threshold_english_description_ascii:
                    ctr_p_skip_no_english += 1
                    continue
                del perc_title, perc_description, full_description

                # ------------------------------------------------------------------------------------------------------
                # Save the current product to the partial catalogue.
                # ------------------------------------------------------------------------------------------------------
                # Remove garbage characters from the title.
                title = str(re.sub(re_ws, " ", title.replace("<br />", "")).strip())
                # Remove garbage characters from the description.
                description = [str(re.sub(re_ws, " ", x.replace("<br />", "")).strip()) for x in description]
                # Remove garbage characters from the bullet point features.
                bullet_point_features = [str(re.sub(re_ws, " ", x.replace("<br />", "")).strip())
                                         for x in bullet_point_features]

                catalogue[parent_asin] = {
                    "parent_asin": parent_asin,
                    "title": title,
                    "description": description + bullet_point_features,
                    "details": details,
                    "categories": categories,
                    "store": store,
                    "price": price,
                    "num_ratings": num_rating,
                    "avg_ratings": avg_rating,
                    "valid_reviews": [],
                    "valid_ratings": { 1: 0, 2: 0, 3: 0, 4: 0, 5: 0 },
                    "num_valid_ratings": 0,
                    "avg_valid_ratings": 0.0
                }
                del parent_asin, title, categories, description, bullet_point_features, details, store, price
                del avg_rating, num_rating, bought_together
            del line
        del fi

        # --------------------------------------------------------------------------------------------------------------
        # Process the "review" file currently being considered.
        # --------------------------------------------------------------------------------------------------------------
        with gzip.open(f"{args.reviews_folder}{os.sep}{cmr}.jsonl.gz", "rt", encoding="utf-8") as fi:
            for line in fi:
                # ------------------------------------------------------------------------------------------------------
                # Extract the data from the jsonl record.
                # ------------------------------------------------------------------------------------------------------
                data = json.loads(line)
                parent_asin = data["parent_asin"]
                title = data["title"]
                text = data["text"]
                timestamp = data["timestamp"]
                user_id = data["user_id"]
                verified_purchase = data["verified_purchase"]
                rating = data["rating"]
                helpful = data["helpful_vote"]
                del data

                # ------------------------------------------------------------------------------------------------------
                # Check if the review record should be discarded.
                # ------------------------------------------------------------------------------------------------------
                # 1) Discard the review if it has an invalid asin.
                if parent_asin is None or (not isinstance(parent_asin, str)) or len(parent_asin) <= 0:
                    ctr_r_skip_no_asin += 1
                    continue
                # 2) Discard the review if it does not belong to any products selected.
                if not parent_asin in catalogue:
                    ctr_r_skip_no_product += 1
                    continue
                # 3) Discard the review if it doesn't have a user id or is empty.
                if user_id is None or (not isinstance(user_id, str)) or len(user_id) <= 0:
                    ctr_r_skip_no_user += 1
                    continue
                # 4) Discard the review if it doesn't have a timestamp of it is empty.
                if timestamp is None or (not isinstance(timestamp, int)) or timestamp < 0:
                    ctr_r_skip_no_timestamp += 1
                    continue
                # 5) Discard the review if it doesn't have a rating of it is invalid.
                if rating is None or (not isinstance(rating, float)) or (not math.isfinite(rating)) or \
                        rating < 1.0 or rating > 5.0:
                    ctr_r_skip_no_rating += 1
                    continue
                # Cast the rating from float to int.
                rating = int(round(rating + 0.001))
                # Save the current rating into the catalogue data.
                catalogue[parent_asin]["valid_ratings"][rating] += 1
                # 6) Discard the review if it doesn't have a title of it is empty.
                if title is None or (not isinstance(title, str)) or len(title) <= 0:
                    ctr_r_skip_no_title += 1
                    continue
                # 7) Discard the review if it doesn't have a text of it is empty.
                if text is None or (not isinstance(text, str)) or len(text) <= 0:
                    ctr_r_skip_no_text += 1
                    continue
                # 8) Discard the review if it doesn't have a verified_purchase of it is empty.
                if verified_purchase is None or (not isinstance(verified_purchase, bool)):
                    ctr_r_skip_no_verified += 1
                    continue
                # 9) Discard the review if it doesn't have a helpful of it is invalid.
                if helpful is None or (not isinstance(helpful, int)) or helpful < 0:
                    ctr_r_skip_no_helpful += 1
                    continue
                # 10) Discard the review if the text is too short.
                if len(text) < args.threshold_too_short_description:
                    ctr_r_skip_no_text += 1
                    continue
                # 11) Discard the review if the text is likely not in english language.
                perc_title = sum(1 for ch in title if ord(ch) <= 255) / len(title)
                perc_text = sum(1 for ch in text if ord(ch) <= 255) / len(text)

                if perc_title < args.threshold_english_title_ascii or \
                    perc_text < args.threshold_english_description_ascii:
                    ctr_r_skip_no_english += 1
                    continue
                del perc_title, perc_text

                # Remove garbage characters from the title.
                title = str(re.sub(re_ws, " ", title.replace("<br />", "")).strip())
                # Remove garbage characters from the text.
                text = str(re.sub(re_ws, " ", text.replace("<br />", "")).strip())

                catalogue[parent_asin]["valid_reviews"].append({
                    "user": user_id,
                    "timestamp": timestamp,
                    "verified purchase": verified_purchase,
                    "title": title,
                    "text": text,
                    "rating": rating,
                    "helpful": helpful
                })

                del parent_asin, user_id, timestamp, verified_purchase, title, text, rating, helpful
            del line
        del fi

        del re_ws

        # --------------------------------------------------------------------------------------------------------------
        # Fix "num_valid_ratings" e "avg_valid_ratings" for each product of the catalogue.
        # --------------------------------------------------------------------------------------------------------------
        for k, v in catalogue.items():
            # Sort the reviews by timestamp, in ascending order.
            v["valid_reviews"] = sorted(v["valid_reviews"], key=lambda x: x["timestamp"], reverse=False)

            # Compute the average rating among those extracted from all product reviews.
            sum_ratings = sum(k1 * v1 for k1, v1 in v["valid_ratings"].items())
            num_ratings = len(v["valid_ratings"])
            avg_ratings = sum_ratings / num_ratings if num_ratings > 0 else None
            del sum_ratings

            # Fix "num_valid_ratings" e "avg_valid_ratings".
            catalogue[k]["num_valid_ratings"] = num_ratings
            catalogue[k]["avg_valid_ratings"] = avg_ratings
            del num_ratings, avg_ratings
        # del k, v

        # --------------------------------------------------------------------------------------------------------------
        # Append the current partial catalogue data to the output file on disk.
        # --------------------------------------------------------------------------------------------------------------
        if args.output_filename.endswith(".gz"):
            fo = gzip.open(args.output_filename, "at", encoding="utf-8")
        else:
            fo = open(args.output_filename, "at", encoding="utf-8")

        with fo:
            for _, v in sorted(catalogue.items(), key=operator.itemgetter(0), reverse=False):
                print(json.dumps(v), file=fo, flush=True)
        del fo

        ctr_valid_products += len(catalogue)

        ctr += 1
        print(f"Split {ctr} / {len(common)} done - \"{cmr}\"!", flush=True)
    del cmr

    # ------------------------------------------------------------------------------------------------------------------
    # Output the final statistics.
    # ------------------------------------------------------------------------------------------------------------------
    print("\nDone!\n", flush=True)
    print(f"Number of valid products: {ctr_valid_products}", flush=True)
    print(f"Number of products WITH bought together: {ctr_bought_together_yes}", flush=True)
    print(f"Number of products WITHOUT bought together: {ctr_bought_together_no}", flush=True)
    print("", flush=True)
    print(f"Number of skipped products, due to missing ASIN: {ctr_p_skip_no_asin}", flush=True)
    print(f"Number of skipped products, due to missing title: {ctr_p_skip_no_title}", flush=True)
    print(f"Number of skipped products, due to missing description and bullet point list: {ctr_p_skip_no_description}", flush=True)
    print(f"Number of skipped products, due to missing categories: {ctr_p_skip_no_categories}", flush=True)
    print(f"Number of skipped products, due to missing store: {ctr_p_skip_no_store}", flush=True)
    print(f"Number of skipped products, due to missing price: {ctr_p_skip_no_price}", flush=True)
    print(f"Number of skipped products, due to not being in english: {ctr_p_skip_no_english}", flush=True)
    print("", flush=True)
    print(f"Number of skipped reviews, due to missing ASIN: {ctr_r_skip_no_asin}", flush=True)
    print(f"Number of skipped reviews, due to missing title: {ctr_r_skip_no_title}", flush=True)
    print(f"Number of skipped reviews, due to missing text: {ctr_r_skip_no_text}", flush=True)
    print(f"Number of skipped reviews, due to missing user: {ctr_r_skip_no_user}", flush=True)
    print(f"Number of skipped reviews, due to missing timestamp: {ctr_r_skip_no_timestamp}", flush=True)
    print(f"Number of skipped reviews, due to missing verified purchase: {ctr_r_skip_no_verified}", flush=True)
    print(f"Number of skipped reviews, due to missing rating: {ctr_r_skip_no_rating}", flush=True)
    print(f"Number of skipped reviews, due to missing helpful: {ctr_r_skip_no_helpful}", flush=True)
    print(f"Number of skipped reviews, due to not being in english: {ctr_r_skip_no_english}", flush=True)


if __name__ == '__main__':
    main()
