import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
 
# Download NLTK resources (if you haven't done this already)
nltk.download('stopwords')
nltk.download('wordnet')
 
# Load data from CSV
df = pd.read_csv('C:/Users/sneha.khandelwal/Downloads/archive/ecommerce_intent_dataset.csv')
 
# Display the first few rows to understand the structure
print(df.head())
 
# Data cleaning and preprocessing functions
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text
 
def preprocess_text(text):
    text = clean_text(text)
    stop_words = set(stopwords.words('english')) - {'order', 'track', 'status'}  # Keep important words
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)
 
# Apply preprocessing to the 'User Query' column
df['user_query'] = df['User Query'].apply(preprocess_text)
 
# Vectorization
vectorizer = TfidfVectorizer(ngram_range=(1, 2))  # Using unigrams and bigrams
X = vectorizer.fit_transform(df['user_query'])
y = df['Intent']
 
# Splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
 
# Model Training with Logistic Regression
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
 
# Evaluate the model
y_pred = model.predict(X_test)
print(f"\nLogistic Regression Classification Report:\n", classification_report(y_test, y_pred))
 
# Function to predict intents for new queries
def predict_intent(new_queries):
    cleaned_queries = [preprocess_text(query) for query in new_queries]
    X_new = vectorizer.transform(cleaned_queries)
    preds = model.predict(X_new)
    return preds
 
# Sample new user queries for intent prediction
new_queries = [
    "What payment options are available?",
    "How do I reset my account password?",
    "Do you have this product in stock?",
    "Where is my order currently?"
]
 
# Predict intents for the new queries
predicted_intents = predict_intent(new_queries)
 
# Display the predicted intents for each query
for query, intent in zip(new_queries, predicted_intents):
    print(f"\nQuery: '{query}'")
    print(f"  Predicted intent: {intent}")
