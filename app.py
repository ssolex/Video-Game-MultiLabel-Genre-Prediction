import streamlit as st
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import re

# Download necessary NLTK data (if you haven't already)
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')


def clean_text(text):
    # Remove backslash-apostrophe
    text = re.sub("\'", "", text)
    # Remove everything except alphabets
    text = re.sub("[^a-zA-Z]", " ", text)
    # Remove extra whitespaces
    text = ' '.join(text.split())
    # Convert text to lowercase
    text = text.lower()
    # Remove stop words
    stop_words = set(stopwords.words('english'))
    # Update the stopwords with additional words
    stop_words.update(['one', 'two', 'first', 'second', 'three', 'four', 'five', 'six',
                       'seven', 'eight', 'nine', 'ten', 'go', 'gets', 'may', 'also', 'across',
                       'among', 'beside', 'like', 'set', 'however', 'yet', 'within', 'last', 'well',
                       'blyton', 'hapq', 'picchu', 'letsplayvideogames', 'pwor', 'uuum', 'zerzura', 'pachomius',
                       'serana', 'perderam', 'zon', 'even'])

    stemmer = SnowballStemmer("english")
    cleaned_text = ' '.join([stemmer.stem(w)
                            for w in text.split() if not w in stop_words])

    return cleaned_text


# Load the trained model and vectorizer
with open('model_lemma_lr.pkl', 'rb') as mf:
    model = pickle.load(mf)

with open('vectorizer.pkl', 'rb') as vf:
    vectorizer = pickle.load(vf)

with open('mlb.pkl', 'rb') as mlbf:
    mlb = pickle.load(mlbf)


def predict_genre(q):
    q = clean_text(q)
    q_vec = vectorizer.transform([q])
    q_pred = model.predict(q_vec)
    result = mlb.inverse_transform(q_pred)
    if len(result[0]) == 0:
        return "Sorry! Unable to p redict a Genre"
    else:
        return result[0]


def main():
    # Streamlit app
    st.title('Game Genre Classifier')

    # Input from the user
    user_input = st.text_area("Enter the game's description:", height=300)

    if st.button('Predict Genre',):
        predicted_genres = predict_genre(user_input)
        st.success(f'The predicted genre is: {predicted_genres}')


if __name__ == '__main__':
    main()
