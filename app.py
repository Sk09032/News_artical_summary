import streamlit as st
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.cli import download
from heapq import nlargest
import newspaper
from newspaper import Article
import string
download("en_core_web_sm")
nlp = spacy.load("en_core_web_sm")
## dnkd
list_of_stopwords = list(STOP_WORDS)
punctuation = string.punctuation + '\n'

st.set_page_config(page_title="News Summarization", layout="wide", initial_sidebar_state="collapsed")

st.title("This is a Extractive News Summarization Streamlit App.")
st.subheader("Using python libraries")

url = st.text_input("Enter news article link")
percent = st.number_input("Enter the ratio of summary (0-1)", min_value=0.0, max_value=1.0, value=0.3)



def generate_summary():
    article=Article(url)
    article.download()
    article.parse()

    text=article.text
    if text:
        doc = nlp(text)
        sentence_tokens = [sent for sent in doc.sents]
        
        word_freq = {}
        for word in doc:
            if word.text.lower() not in list_of_stopwords and word.text not in punctuation:
                word_freq[word.text] = word_freq.get(word.text, 0) + 1

        max_freq = max(word_freq.values(), default=1)
        for word in word_freq.keys():
            word_freq[word] /= max_freq

        sent_score = {}
        for sent in sentence_tokens:
            for word in sent:
                if word.text in word_freq.keys():
                    sent_score[sent] = sent_score.get(sent, 0) + word_freq[word.text]

        select_length = max(1, int(len(sentence_tokens) * percent))
        summary = nlargest(select_length, sent_score, key=sent_score.get)

        final_summary = [word.text for word in summary]
        summary_text = ' '.join(final_summary)
        st.markdown("## News Article Title:")
        st.markdown(f"#### {article.title}")
        st.write("Summary:")
        st.write(summary_text)

if st.button("Generate Summary"):
    generate_summary()
