import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Load NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Initialize the Porter Stemmer
ps = PorterStemmer()

def transform_text(text):
    """ Preprocess the input text for prediction. """
    text = text.lower()
    text = nltk.word_tokenize(text)

    # Remove non-alphanumeric tokens
    text = [i for i in text if i.isalnum()]

    # Remove stopwords and punctuation
    text = [i for i in text if i not in stopwords.words('english') and i not in string.punctuation]

    # Apply stemming
    text = [ps.stem(i) for i in text]

    return " ".join(text)

# Load models
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# Set up the Streamlit app
st.set_page_config(page_title="Email/SMS Spam Classifier", page_icon="📩", layout="wide")

# Apply custom CSS for styling
st.markdown("""
    <style>
    body {
        background-image: linear-gradient(to bottom right, rgba(34,193,195,0.6), rgba(253,187,45,0.6)), url("https://repository-images.githubusercontent.com/275336521/20d38e00-6634-11eb-9d1f-6a5232d0f84f");
        background-size: cover;
        background-attachment: fixed;
        color: #f0f0f0;
        font-family: 'Roboto', sans-serif;
    }
    .stTextArea textarea {
        border-radius: 10px;
        padding: 10px;
        border: 1px solid #ddd;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        resize: none;
    }
    .stButton button {
        background-color: #22c1c3;
        color: white;
        border-radius: 12px;
        padding: 0.5em 1em;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        font-size: 16px;
        cursor: pointer;
    }
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.3);
        background-color: #1aa0a2;
    }
    .stHeader h1 {
        color: #ffdb58;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    footer {
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        background-color: rgba(0, 0, 0, 0.5);
        color: #f0f0f0;
        text-align: center;
        padding: 1em 0;
    }
    </style>
""", unsafe_allow_html=True)

# Title of the app
st.title("📩 Email/SMS Spam Classifier")

# Input field for the message
input_sms = st.text_area("Enter the message:", height=150)

# Predict button
if st.button('Predict'):
    if input_sms:
        # 1. Preprocess the input
        transformed_sms = transform_text(input_sms)
        # 2. Vectorize the preprocessed text
        vector_input = tfidf.transform([transformed_sms])
        # 3. Predict using the loaded model
        result = model.predict(vector_input)[0]
        # 4. Display result
        if result == 1:
            st.subheader("Prediction: Spam")
        else:
            st.subheader("Prediction: Not Spam")
    else:
        st.warning("Please enter a message to classify.")

# Footer
st.markdown("""
    <footer>
        <p>© 2024 Email/SMS Spam Classifier | Created by Ankit Singh</p>
    </footer>
""", unsafe_allow_html=True)
