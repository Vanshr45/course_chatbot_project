r why).
"""
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import joblib
import nltk
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

# ---- Load the trained retrieval model ----
bundle = joblib.load('chatbot_retrieval.joblib')
vectorizer = bundle["vectorizer"]
question_matrix = bundle["question_matrix"]
answers = bundle["answers"]


def preprocess(text: str) -> str:
    return ' '.join(nltk.word_tokenize(str(text).lower()))


def get_response(question: str, threshold: float = 0.25) -> str:
    q_proc = preprocess(question)
    q_vec = vectorizer.transform([q_proc])
    sims = cosine_similarity(q_vec, question_matrix)[0]
    best_idx = np.argmax(sims)
    if sims[best_idx] < threshold:
        return "Sorry, I don't know the answer to that yet. Try rephrasing your question."
    return answers[best_idx]


# ---- Dash app ----
app = dash.Dash(__name__)
app.title = "Chatbot"

app.layout = html.Div(
    style={"maxWidth": "700px", "margin": "40px auto", "fontFamily": "Arial, sans-serif"},
    children=[
        html.H1("Chatbot", style={'textAlign': 'center'}),
        dcc.Textarea(
            id='user-input',
            value='Type your question here...',
            style={'width': '100%', 'height': 100}
        ),
        html.Button('Submit', id='submit-button', n_clicks=0, style={"marginTop": "10px"}),
        html.Div(id='chatbot-output', style={'padding': '10px'})
    ]
)


@app.callback(
    Output('chatbot-output', 'children'),
    Input('submit-button', 'n_clicks'),
    [State('user-input', 'value')]
)
def update_output(n_clicks, user_input):
    if n_clicks and n_clicks > 0 and user_input and user_input.strip():
        answer = get_response(user_input)
        return html.Div([
            html.P(f"You: {user_input}", style={'margin': '10px'}),
            html.P(
                f"Bot: {answer}",
                style={'margin': '10px', 'backgroundColor': '#f0f0f0', 'padding': '10px'}
            ),
        ])
    return ''


if __name__ == '__main__':
    app.run(debug=True)
