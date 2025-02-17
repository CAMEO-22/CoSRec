# CoSRec
Conversational Search and Recommendation Dataset



## Repository Structure

```
\
├── dataset
│   ├── crowd
│   │   ├── quality.jsonl
│   │   ├── conversations.jsonl
│   │   ├── keywords.jsonl
│   │   ├── intent_annotations.jsonl
│   │   └── profiles.jsonl
│   ├── raw
│   │   └── conversations.jsonl
│   └── curated
│       ├── quality.jsonl
│       ├── intents.jsonl
│       ├── qrels.qrels
│       ├── conversations.jsonl
│       ├── keywords.jsonl
│       └── profiles.jsonl
├── README.md: 
├── scripts
│   └── catalogue_preprocessing.py
└── prompts
    ├── user_summary_prompt.txt
    ├── user_keywords_prompt.txt
    ├── product_to_query_prompt.txt
    └── conversation_generation_prompt.txt
```
This repository is structured as follows:
- dataset: holds the files representing the CoSRec dataset.
    - raw: 8938  non-annotated conversations
    - crowd: 291  annotated conversations
    - curated: 20 deeply annotated conversations
- scripts: holds the scripts needed to process the Amazon Reviews dataset and obtain its filtered version (AR-filtered)
- prompts: holds the prompts used for  generating the conversations ("conversation_generation_prompt.txt"), extracting a search-like query from a product ("product_to_query_prompt.txt"), extracting the user profile summary ("user_summary_prompt.txt") and keywords ("user_keywords_prompt.txt") from the reviews.

Moreover, each partition of CoSRec (Raw, Crowd and Curated) in the dataset folder contains several files:
- conversations.jsonl: file containing the conversations, associated to some IDs
- quality.jsonl: quality ratings of the conversations
- intents.jsonl: manually parsed intents associated to each user utterance of each conversation (Curated Only)
- intents_annotations.jsonl: raw labelled intents associated to each user utterance of each conversation (Crowd Only)
- profiles.json: user profile summaries associated to each conversation.
- keywords.json: user profile keywords associated to each conversation.
- qrels.json: relevance judgments in TREC-style format

