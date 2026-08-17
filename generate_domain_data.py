"""
Generates course-specific Question/Answer pairs for a Data Science / AI course chatbot.
These mirror the kind of module-referencing answers shown in the course demo
(e.g. "This is covered in the Data Manipulation and Analysis with NumPy module.").
"""
import pandas as pd

DOMAIN_QA = [
    ("What is NumPy?", "NumPy is a Python library for fast numerical computing, built around powerful n-dimensional arrays. This is covered in the Data Manipulation and Analysis with NumPy module."),
    ("What is a NumPy array?", "NumPy arrays are a powerful data structure in Python for numerical computations. This is covered in the Data Manipulation and Analysis with NumPy module."),
    ("How do I create a NumPy array?", "You create one with np.array(), e.g. np.array([1, 2, 3]). See the Data Manipulation and Analysis with NumPy module for examples."),
    ("What is pandas?", "pandas is a Python library for working with tabular data using DataFrames and Series. This is covered in the Data Manipulation with pandas module."),
    ("What is a pandas DataFrame?", "A DataFrame is a 2-dimensional labeled data structure in pandas, similar to a spreadsheet or SQL table."),
    ("How do I read a CSV file in pandas?", "Use pd.read_csv('filename.csv') to load a CSV file into a pandas DataFrame."),
    ("What is Matplotlib?", "Matplotlib is a Python library for creating static, animated, and interactive visualizations and charts."),
    ("What is Seaborn?", "Seaborn is a Python visualization library based on Matplotlib that provides a high-level interface for drawing attractive statistical graphics."),
    ("How do I make a bar chart in Seaborn?", "Use sns.barplot(x=..., y=..., data=df) to create a bar chart with Seaborn."),
    ("What is scikit-learn?", "scikit-learn is a Python library that provides simple, efficient tools for machine learning, including classification, regression, and clustering."),
    ("What is a machine learning pipeline?", "A pipeline in scikit-learn chains together preprocessing steps and a model, e.g. make_pipeline(TfidfVectorizer(), MultinomialNB()), so data flows through each step automatically."),
    ("What is TF-IDF?", "TF-IDF (Term Frequency-Inverse Document Frequency) is a technique that converts text into numerical values reflecting how important a word is to a document relative to a collection of documents."),
    ("What does TfidfVectorizer do?", "TfidfVectorizer converts a collection of text documents into a matrix of TF-IDF features, turning words into numbers a model can learn from."),
    ("What is tokenization?", "Tokenization is the process of splitting text into individual units called tokens, usually words or subwords, as a first step in NLP preprocessing."),
    ("What is nltk.word_tokenize used for?", "nltk.word_tokenize() splits a string of text into a list of individual word tokens."),
    ("What are stop words?", "Stop words are common words like 'the', 'is', and 'a' that are often removed during text preprocessing because they carry little useful meaning for a model."),
    ("What is stemming?", "Stemming reduces a word to its root form by stripping suffixes, e.g. 'running' becomes 'run', though the result isn't always a real word."),
    ("What is lemmatization?", "Lemmatization reduces a word to its dictionary base form (lemma) using vocabulary and grammar rules, e.g. 'better' becomes 'good'."),
    ("What is NLP?", "NLP (Natural Language Processing) is a field of AI focused on enabling computers to understand, interpret, and generate human language."),
    ("What is a Naive Bayes classifier?", "Naive Bayes is a probabilistic classification algorithm based on Bayes' theorem that assumes features are independent; it works well for text classification tasks."),
    ("What is MultinomialNB?", "MultinomialNB is a Naive Bayes classifier variant suited for classification with discrete features, commonly used for text data represented as word counts or TF-IDF scores."),
    ("What is train_test_split?", "train_test_split() from scikit-learn randomly splits a dataset into training and testing subsets so you can evaluate a model on unseen data."),
    ("What is make_pipeline?", "make_pipeline() from scikit-learn chains preprocessing and modeling steps into a single object, so calling .fit() and .predict() runs every step in order."),
    ("What is model.fit used for?", "model.fit(X_train, y_train) trains a machine learning model on the provided training data."),
    ("What is model.predict used for?", "model.predict() uses a trained model to generate predictions for new, unseen input data."),
    ("What is Dash?", "Dash is a Python framework for building interactive web applications and dashboards without needing to write JavaScript."),
    ("What is Plotly?", "Plotly is a graphing library used to create interactive charts and visualizations, often used together with Dash."),
    ("How do I initialize a Dash app?", "You initialize a Dash app with: import dash; app = dash.Dash(__name__)."),
    ("How do I define a Dash layout?", "You define the layout using app.layout = html.Div([...]) with Dash HTML and core components like html.H1, dcc.Input, and html.Button."),
    ("What is a Dash callback?", "A callback is a Python function decorated with @app.callback() that automatically runs and updates the app's output whenever a specified input changes."),
    ("What does dcc.Textarea do?", "dcc.Textarea is a Dash core component that renders a multiline text input box for the user to type into."),
    ("What does html.Button do?", "html.Button renders a clickable button in a Dash app's layout; its n_clicks property increments each time it's clicked."),
    ("How do I run a Dash app?", "You run a Dash app by calling app.run(debug=True) inside an if __name__ == '__main__': block."),
    ("What is State in a Dash callback?", "State lets a callback read a component's current value without triggering the callback itself; only the Input component triggers the callback."),
    ("What is joblib used for?", "joblib is used to save and load trained Python objects like machine learning models efficiently, e.g. joblib.dump(model, 'model.joblib')."),
    ("What is vectorization in NLP?", "Vectorization is the process of converting text into numerical vectors so machine learning models can process it, e.g. via TF-IDF or word embeddings."),
    ("What is X.shape in scikit-learn?", "X.shape returns the dimensions of a matrix, e.g. (48, 112) means 48 samples and 112 features."),
    ("What is a corpus in NLP?", "A corpus is a large, structured collection of text used for training or analyzing language models."),
    ("What is a bag of words model?", "Bag of words represents text as an unordered collection of word counts, ignoring grammar and word order but keeping multiplicity."),
    ("What is overfitting?", "Overfitting happens when a model learns the training data too closely, including its noise, and performs poorly on new, unseen data."),
    ("What is accuracy in machine learning?", "Accuracy is the proportion of correct predictions a model makes out of all predictions."),
    ("What is a feature in machine learning?", "A feature is an individual measurable input variable used by a model to make predictions."),
    ("What is supervised learning?", "Supervised learning is a machine learning approach where a model learns from labeled input-output pairs to predict outputs for new inputs."),
    ("What is classification in machine learning?", "Classification is a supervised learning task where a model predicts a discrete category or label for given input data."),
    ("What is a DataFrame column?", "A DataFrame column is a pandas Series representing one labeled field of data, e.g. data['Question']."),
    ("How do I install a Python package?", "You install a Python package using pip, e.g. pip install pandas."),
    ("What is a virtual environment?", "A virtual environment is an isolated Python environment that lets you install packages for a specific project without affecting your system Python."),
    ("What does random_state do in train_test_split?", "random_state sets a seed for the random split so the same train/test split is reproduced every time the code runs."),
    ("What is the difference between fit and fit_transform?", "fit() learns parameters from data, while fit_transform() learns the parameters and immediately applies the transformation, returning the transformed data."),
    ("What port does a Dash app run on by default?", "A Dash app runs on http://127.0.0.1:8050 by default when started with app.run(debug=True)."),
]

def get_domain_dataframe():
    return pd.DataFrame(DOMAIN_QA, columns=["Question", "Answer"])

if __name__ == "__main__":
    df = get_domain_dataframe()
    print(f"Generated {len(df)} domain-specific Q&A pairs")
    print(df.head())
