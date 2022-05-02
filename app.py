import streamlit as st
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from transformers.pipelines import pipeline

st.cache(show_spinner=False)
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("twmkn9/distilbert-base-uncased-squad2")
    model = AutoModelForQuestionAnswering.from_pretrained("twmkn9/distilbert-base-uncased-squad2")
    nlp_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer)
    return nlp_pipeline

nlp_pipeline = load_model()

st.header("Reginald von Loom")
st.text("Your personal legal and financial assistant")

add_text_sidebar = st.sidebar.title("Menu")
add_text_sidebar = st.sidebar.text("Random Text")

question = st.text_input(label="Insert a question")
text = st.text_area(label="Context")

if (not len(text) == 0) and (not len(question) == 0):
    x_dict = nlp_pipeline(context=text, question=question)
    st.text("Answer: ", x_dict["answer"])