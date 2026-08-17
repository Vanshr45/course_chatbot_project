
import os
import subprocess
import glob
import yaml
import pandas as pd

from generate_domain_data import get_domain_dataframe

CORPUS_DIR = "chatterbot-corpus"
CORPUS_URL = "https://github.com/gunthercox/chatterbot-corpus.git"
ENGLISH_DIR = os.path.join(CORPUS_DIR, "chatterbot_corpus", "data", "english")


def ensure_corpus():
    if not os.path.isdir(ENGLISH_DIR):
        print("Cloning chatterbot-corpus ...")
        subprocess.run(["git", "clone", "--depth", "1", CORPUS_URL, CORPUS_DIR], check=True)
    else:
        print("chatterbot-corpus already present, skipping clone.")


def parse_corpus():
    pairs = []
    yml_files = sorted(glob.glob(os.path.join(ENGLISH_DIR, "*.yml")))
    for path in yml_files:
        name = os.path.basename(path)
        with open(path, "r", encoding="utf-8") as f:
            content = yaml.safe_load(f)
        if not content or "conversations" not in content:
            continue
        for convo in content["conversations"]:
            # Pair up consecutive turns: (0,1), (2,3), ...
            for i in range(0, len(convo) - 1, 2):
                q, a = convo[i], convo[i + 1]
                if isinstance(q, str) and isinstance(a, str):
                    pairs.append((q.strip(), a.strip(), name.replace(".yml", "")))
    df = pd.DataFrame(pairs, columns=["Question", "Answer", "category"])
    print(f"Parsed {len(df)} pairs from {len(yml_files)} category files.")
    return df


def build():
    ensure_corpus()
    corpus_df = parse_corpus()
    domain_df = get_domain_dataframe()
    domain_df["category"] = "course_data_science"

    merged = pd.concat([corpus_df, domain_df], ignore_index=True)
    merged["Question"] = merged["Question"].astype(str).str.strip()
    merged["Answer"] = merged["Answer"].astype(str).str.strip()

    # Drop empties and parsing artifacts (e.g. letter-spelling games in the
    # source corpus produce single-character "Q -> A" pairs that are noise,
    # not real question/answer content).
    merged = merged[
        (merged["Question"].str.len() >= 3) &
        (merged["Answer"].str.len() >= 2)
    ]
    merged = merged.drop_duplicates(subset=["Question", "Answer"]).reset_index(drop=True)

    print(f"Final merged dataset: {len(merged)} rows")
    print(merged["category"].value_counts())

    merged[["Question", "Answer"]].to_csv("chatbot_dataset.csv", index=False)
    print("Saved chatbot_dataset.csv")
    return merged


if __name__ == "__main__":
    build()
