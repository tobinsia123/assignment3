import re
import json
import math
import math
from collections import defaultdict
from nltk.stem import PorterStemmer
import json


with open("inverted_index.json", "r", encoding="utf-8") as f:
    final_index = json.load(f)

with open("doc_id_map.json", "r", encoding="utf-8") as f:
    doc_id_map = json.load(f)

def print_best_5_postings(query_tokens, final_index, total_docs=None):
    if total_docs is None:
        total_docs = 0
        for term_data in final_index.values():
            for posting in term_data["postings"]:
                total_docs = max(total_docs, int(posting["doc_id"]))

    scores = defaultdict(float)

    for term in query_tokens:
        if term not in final_index:
            continue

        df = final_index[term]["df"]

        idf = math.log((total_docs + 1) / (df + 1)) + 1

        for posting in final_index[term]["postings"]:
            doc_id = posting["doc_id"]
            tf = posting["tf"]

            scores[doc_id] += tf * idf

    best_5 = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:5]

    for rank, (doc_id, score) in enumerate(best_5, start=1):
        print(f"{rank}. doc={doc_id_map[doc_id]}, score={score:.4f}")

    return best_5   
stemmer = PorterStemmer()
def tokenize(text):
        tokens = re.findall(r"[a-zA-Z0-9]+", text.lower())
        return [stemmer.stem(token) for token in tokens]

while True:
    query = input("Enter your search query (or 'exit' to quit): ")
    if query.lower() == "exit":
        break
    print_best_5_postings(tokenize(query), final_index)