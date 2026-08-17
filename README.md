# Course Chatbot — TF-IDF Q&A Chatbot with Dash UI

A text-classification-style chatbot trained on a large Question/Answer dataset,
served through a Dash web app. Same structure as the course tutorial
(preprocess → TF-IDF vectorize → train → `get_response()` → Dash textarea UI),
built on a much larger real dataset (1,059 rows vs. the original 48).

## Setup

```bash
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
```

## Rebuild the dataset (optional — chatbot_dataset.csv is already included)

```bash
python build_dataset.py
```
This clones the open-source [chatterbot-corpus](https://github.com/gunthercox/chatterbot-corpus)
(BSD-licensed, ~2,000 conversational Q&A pairs across 21 topics — AI, science,
history, literature, trivia, psychology, computers, etc.), cleans out parsing
noise, and merges it with 50 hand-written data-science/course Q&A pairs
(NumPy, pandas, Seaborn, TF-IDF, Naive Bayes, Dash, etc.) from
`generate_domain_data.py`.

## Train the model

```bash
python train_model.py
```
This trains **two** models, both TF-IDF based:

1. **Naive Bayes classifier** (`chatbot_model_naive_bayes.joblib`) — the exact
   approach from the tutorial (`make_pipeline(TfidfVectorizer(), MultinomialNB())`).
   Included for fidelity to the course material. Its accuracy is low on this
   dataset because almost every question has a unique answer, so there's
   nothing for a *classifier* to generalize from — the same issue visible in
   the original tutorial video, where `get_response("What is NLP?")` returned
   an unrelated Seaborn answer.
2. **TF-IDF + cosine-similarity retrieval** (`chatbot_retrieval.joblib`) — finds
   the closest matching known question and returns its answer. This is the
   standard working approach for small/medium Q&A chatbots and is what
   `chatbot_app.py` actually uses.

## Run the app

```bash
python chatbot_app.py
```
Open **http://127.0.0.1:8050** in your browser, type a question into the
textarea, and click Submit.

## Files

| File | Purpose |
|---|---|
| `chatbot_dataset.csv` | Final merged Question/Answer dataset (1,059 rows) |
| `generate_domain_data.py` | Hand-written course/data-science Q&A pairs |
| `build_dataset.py` | Downloads + parses chatterbot-corpus, merges, cleans, saves CSV |
| `train_model.py` | Preprocessing, TF-IDF, Naive Bayes training, retrieval model training |
| `chatbot_model_naive_bayes.joblib` | Trained Naive Bayes pipeline (tutorial-exact) |
| `chatbot_retrieval.joblib` | TF-IDF vectorizer + question matrix + answers (used by the app) |
| `chatbot_app.py` | Dash web app serving the chatbot |
| `requirements.txt` | Python dependencies |

## Notes

- `app.run()` is used instead of the deprecated `app.run_server()`.
- To extend the bot's course-specific knowledge, add rows to
  `generate_domain_data.py` and re-run `build_dataset.py` + `train_model.py`.
