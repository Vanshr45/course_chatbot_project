"""
Trains the chatbot on chatbot_dataset.csv.

Includes TWO approaches, both TF-IDF based:

1. NAIVE BAYES CLASSIFIER (matches the course tutorial exactly).
   Treats every unique Answer as a class. This works fine on a tiny, curated
   dataset where each answer repeats for paraphrased questions, but with a
   large open dataset almost every question has a one-of-a-kind answer, so a
   classifier has nothing to generalize from and accuracy is poor. Included
   here for fidelity to the course material, and printed honestly.

2. TF-IDF + COSINE-SIMILARITY RETRIEVAL (what the Dash app actually uses).
   Finds the most similar known question to the user's input and returns its
   answer. This is the standard, working approach for small-to-medium Q&A
   chatbots and performs far better in practice.
"""
import pandas as pd
import nltk
import joblib
import numpy as np

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity


def preprocess(text: str) -> str:
    return ' '.join(nltk.word_tokenize(str(text).lower()))


def main():
    # ---- Load & preprocess (same as the notebook) ----
    data = pd.read_csv('chatbot_dataset.csv')
    print(f"Loaded dataset: {data.shape}")

    data['Question'] = data['Question'].apply(preprocess)

    # ---- Vectorizing Text Data ----
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(data['Question'])
    print(f"TF-IDF matrix shape: {X.shape}")

    # ================================================================
    # Approach 1: Naive Bayes classifier (tutorial-exact code)
    # ================================================================
    X_train, X_test, y_train, y_test = train_test_split(
        data['Question'], data['Answer'], test_size=0.2, random_state=42
    )
    nb_model = make_pipeline(TfidfVectorizer(), MultinomialNB())
    nb_model.fit(X_train, y_train)

    train_acc = nb_model.score(X_train, y_train)
    test_acc = nb_model.score(X_test, y_test)
    print(f"[Naive Bayes] Train accuracy: {train_acc:.3f} | Test accuracy: {test_acc:.3f}")
    print("[Naive Bayes] note: low accuracy is expected here since almost every "
          "question maps to a unique answer class (see docstring above).")

    joblib.dump(nb_model, 'chatbot_model_naive_bayes.joblib')

    # ================================================================
    # Approach 2: TF-IDF + cosine similarity retrieval (used by the app)
    # ================================================================
    retrieval_vectorizer = TfidfVectorizer()
    question_matrix = retrieval_vectorizer.fit_transform(data['Question'])
    answers = data['Answer'].tolist()
    original_questions = data['Question'].tolist()

    def get_response(question: str, threshold: float = 0.25):
        q_proc = preprocess(question)
        q_vec = retrieval_vectorizer.transform([q_proc])
        sims = cosine_similarity(q_vec, question_matrix)[0]
        best_idx = np.argmax(sims)
        if sims[best_idx] < threshold:
            return "Sorry, I don't know the answer to that yet. Try rephrasing your question."
        return answers[best_idx]

    # Quick sanity tests
    for q in ["What is NLP?", "what is numpy", "hello", "What is TF-IDF?", "who are you"]:
        print(f"Q: {q!r}  ->  A: {get_response(q)!r}")

    joblib.dump({
        "vectorizer": retrieval_vectorizer,
        "question_matrix": question_matrix,
        "answers": answers,
        "questions": original_questions,
    }, 'chatbot_retrieval.joblib')
    print("Saved chatbot_model_naive_bayes.joblib and chatbot_retrieval.joblib")


if __name__ == "__main__":
    main()
