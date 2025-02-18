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
- prompts: holds the prompts used for  generating the conversations, extracting a search-like query from a product, extracting the user profile summary and keywords from the reviews.


### Dataset

This directory holds the files constituting the CoSRec dataset, which is splitted in 3 partitions: CoSRec Raw, CoSRec Crowd, CoSRec Curated.

#### CoSRec Raw
CoSRec Raw is a set of 8938 non-annotated conversations.  

The file **conversations.jsonl** contains a conversation for each line. Each conversation corresponds to a dictionary with a single key, the conversation ID, and a single value, the conversation text. The user's utterances start with "U:" and are separated from the system's utterances, starting with "S:" by means of "\n".

#### CoSRec Crowd

CoSRec Raw is a set of 291 annotated conversations.  

This partitions contains the following files:

-  **conversations.jsonl**: json file containing a conversation for each line. Each conversation corresponds to a dictionary with a single key, the conversation ID, and a single value, the conversation text. The user's utterances start with "U:" and are separated from the system's utterances, starting with "S:" by means of "\n".
-  **quality.jsonl**: json file containing the ratings provided by the annotators during the quality assessment. Each line of the file corresponds to a conversation. Each conversation takes the form of a dictionary with a single key, the conversation ID, and a single value, the list of ratings. The list is composed of dictionaries. Each dictionary corresponds to the ratings of a user and has as keys the quality aspects ('coherence', 'logicality', 'informativeness', 'fluency') and as values the ratings provided by the annotator.
-  **intent_annotations.jsonl**: json file containing the raw human-labelled intents. Each line of the file corresponds to a conversation. Each conversation takes the form of a dictionary with a single key, the conversation ID, and a single value, the list of intents for each utterance. The list is composed of dictionaries. Each dictionary corresponds to an utterance and has two fields: the utterance id ('utterance'), which is the index of the user utterance in the conversation, and the intent list ('intent_annotations'). The intent list is constituted by a sequence of lists, each representing the intents identified for the considered utterance by a single annotator. For each intent we report the type ('type') and the stand-alone formulation ('query').
-  **profiles.jsonl**: json file containing the profile summaries of the users considered for each conversation. Each line of the file corresponds to a conversation. Each conversation takes the form of a dictionary with a single key, the conversation ID, and a single value, a dictionary containing the user profiles summaries. The user profiles summaries dictionary has as keys the IDs of the considered users and as values the summaries of the users' profiles.
-  **keywords.jsonl**: json file containing the profile keywords of the users considered for each conversation. Each line of the file corresponds to a conversation. Each conversation takes the form of a dictionary with a single key, the conversation ID, and a single value, a dictionary containing the user profiles keywords. The user profiles keywords dictionary has as keys the IDs of the considered users and as values the lists of keywords of the user.

#### CoSRec Curated

CoSRec Raw is a set of 20 deeply annotated conversations, where the raw annotator data is manually reviewed by the authors.  

This partitions contains the following files:
-  **conversations.jsonl**: same as CoSRec Crowd.
-  **quality.jsonl**: same as CoSRec Crowd. Each line of the file corresponds to a conversation. Each conversation takes the form of a dictionary with a single key, the conversation ID, and a single value, the list of intents for each utterance. The list is composed of dictionaries. Each dictionary corresponds to an utterance and has two fields: the utterance id ('utterance'), which is the index of the user utterance in the conversation, and the intent list ('intents'). The intent list is constituted by a sequence of dictionaries, each representing a single intents  and reporting: the intent id ('id'), the intent type ('type') and the list of stand-alone formulaitons ('query_variants'). **Note:** this file does NOT contain the personalized recommendation intents, they can be generated by combining this file and the 'keywords.jsonl' files.
-  **intents.jsonl**: json file containing the reviewd human-labelled intents.
-  **profiles.jsonl**: same as CoSRec Crowd.
-  **keywords.jsonl**: same as CoSRec Crowd.
-  **qrels.qrels**: file containing the relevance judgments for all the intents in TREC-style format.

**Note:** the intent id (file 'intents.jsonl') takes the following form: <conversation_id>_<utterance_id>_<index> where: conversation_id is the identifier of the conversation, utterance_id is the identifier of the utterance (index) WITHIN the conversation and <index> is an incremental field (counter) used to differentiate the identifiers of different intents referring to the same utterance.

**Note:** the id of the personalized intents in the qrels file  ('qrels.qrels') takes the following form: <intent_id>#<user_index> where: intent_id is the id of the intent (structured as explained above) and user_index is a reference to the user for which the intent is personalized. In particular, the user_index is an integer number between 0 and 4, which correspond to the index of the user in the list of users employed to personalize the intents of the considered conversation (**ordered by lexical order of the users' identifiers**).

### Scripts

This directory holds the scripts needed to process the Amazon Reviews dataset and obtain its filtered version (AR-filtered).

### Prompts

This directory contains the prompts used in the interaction with LLama 3.1 8B: 

- The prompt used to generate the conversations corresponds to the file **"conversation_generation_prompt.txt"**.
- The prompt used to extract a search-like query from a product (to be used as request for the retrieval of 10 documents from MS Marco v2.1 Passage) corresponds to the file **"product_to_query_prompt.txt"**.
- The prompt used to create the user profile summary from the user's reviews is **"user_summary_prompt.txt"**.
- The prompt used to generate the user profile keywords from the user's reviews is **"user_keywords_prompt.txt"**.

