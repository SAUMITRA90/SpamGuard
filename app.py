import streamlit  as st
import pickle
import string
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()

def transform_text(text):
    text=text.lower()
    text= nltk.word_tokenize(text)

    y=[]
    for i in text:
        if i.isalnum():
          y.append(i)

    text = y[:]
    y.clear()
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text=y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

st.markdown("""
<h1 style='text-align:center'>
🛡️ SpamGuard
</h1>

<p style='text-align:center;font-size:18px'>
Email & SMS Spam Detection using Machine Learning
</p>
""", unsafe_allow_html=True)

input_sms = st.text_area(
    "✉️ Enter Message",
    height=180,
    placeholder="Paste your SMS or Email here..."
)
if st.button("🚀 Analyze Message"):

# Preprocess
  transformed_sms=transform_text(input_sms)
# vectorize
  vector_input=tfidf.transform([transformed_sms])
# predict
  result=model.predict(vector_input)[0]
# probability
  prob = model.predict_proba(vector_input)[0]
  spam_prob = round(prob[1] * 100, 2)

  st.write("### Confidence Score")
  st.progress(int(spam_prob))
  st.write(f"Spam Probability: {spam_prob}%")
# display
  if result==1:
      st.header("🚨 SPAM")
  else:
      st.header("✅NOT SPAM")



st.markdown("---")
st.markdown(
    "<center>Built using TF-IDF Vectorization & Multinomial Naive Bayes</center>",
    unsafe_allow_html=True
)
