import streamlit as st
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.cli import download
from heapq import nlargest
from newspaper import Article
import string

# Ensure the language model is downloaded and loaded
download("en_core_web_sm")
nlp = spacy.load("en_core_web_sm")

# Define constants
list_of_stopwords = list(STOP_WORDS)
punctuation = string.punctuation + '\n'

# Set up Streamlit app
st.set_page_config(page_title="News Summarization", layout="wide", initial_sidebar_state="collapsed")

st.title("Extractive News Summarization Streamlit App")
st.subheader("Using Python Libraries")

# User inputs
url = st.text_input("Enter news article link")
percent = st.number_input("Enter the ratio of summary (0-1)", min_value=0.0, max_value=1.0, value=0.3)

def generate_summary():
    # Validate URL input
    if not url:
        st.error("Please enter a valid URL.")
        return

    try:
        # Download and parse the article
        article = Article(url)
        article.download()
        article.parse()
        text = article.text

        if text:
            # Process text using spaCy
            doc = nlp(text)
            sentence_tokens = [sent for sent in doc.sents]

            # Calculate word frequencies
            word_freq = {}
            for word in doc:
                if word.text.lower() not in list_of_stopwords and word.text not in punctuation:
                    word_freq[word.text] = word_freq.get(word.text, 0) + 1

            max_freq = max(word_freq.values(), default=1)
            for word in word_freq.keys():
                word_freq[word] /= max_freq

            # Score sentences based on word frequencies
            sent_score = {}
            for sent in sentence_tokens:
                for word in sent:
                    if word.text in word_freq.keys():
                        sent_score[sent] = sent_score.get(sent, 0) + word_freq[word.text]

            # Select top sentences for summary
            select_length = max(1, int(len(sentence_tokens) * percent))
            summary = nlargest(select_length, sent_score, key=sent_score.get)

            # Generate summary text
            final_summary = [sent.text for sent in summary]
            summary_text = ' '.join(final_summary)

            # Display title and summary
            st.markdown("## News Article Title:")
            st.markdown(f"#### {article.title}")
            st.write("### Summary:")
            st.write(summary_text)
        else:
            st.error("No text found in the article. Please check the URL.")
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

# Button to trigger summary generation
if st.button("Generate Summary"):
    generate_summary()
