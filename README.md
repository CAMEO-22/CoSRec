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
├── README.md
├── scripts
│   └── catalogue_preprocessing.py
└── prompts
    ├── user_summary_prompt.txt
    ├── user_keywords_prompt.txt
    ├── product_to_query_prompt.txt
    └── conversation_generation_prompt.txt
```
