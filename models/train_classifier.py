import sys
# import libraries
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
import re
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
import pickle


from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, accuracy_score, f1_score, fbeta_score, classification_report
from sklearn.naive_bayes import MultinomialNB


def load_data(database_filepath):
    """
    Load the SQLite database from databse_filepath and tun it into dataframe. 
    Divide the dataframe data into inputs X and labels Y
    
    INPUTS:
        database_filepath: the path of SQLite database containing disaster messages
    
    OUTPUTS:
        X: inputs used for modeling, containing disaster messages
        Y: labels used for modeling, containing categories of disaster messages
        category_names: list of names containing categories
    """
    engine = create_engine('sqlite:///data/CleanedData.db')
    df = pd.read_sql_table('disaster', engine)
    X = df['message']
    Y = df.iloc[:, 4:]
    category_names = list(df.columns[4:])
    return X,Y, category_names


def tokenize(text):
    """
    Clean and tokenize the input texts for modeling. 
    
    INPUTS:
        text: the messages for cleaning and tokenizing
     
    OUTPUTS:
        clean_tokens: the list of cleaned and tokenized words from the input text
    """
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    
    ## Replace urls in texts
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")
        
    ## Word tokenization
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    
    ## Remove Stop Words
    tokens = [w for w in tokens if w not in stopwords.words("english")]
    
    ## Lemmatize the word tokens
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
        
    return clean_tokens


def build_model():
    """
    Build the pipeline that vectorize and transform tokenized words. 
    Use grid search to find optimal parameters for Random Forest Classifier

    OUTPUT:
        cv: the model with parameters by grid search
    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    parameters = {
        'clf__estimator__min_samples_split': [2, 4],    
    }

    cv = GridSearchCV(estimator = pipeline, param_grid = parameters)
    return cv

def evaluate_model(model, X_test, Y_test, category_names):
    """
    Get the classification report of the model given X_test, Y_test and category_names
    
    INPUTS:
        model: the model for classification
        X_test: the inputs of test data
        Y_test: the labels of test data
        category_names: the list of names of all message categories
    """
    y_pred = model.predict(X_test)
    y_pred_df = pd.DataFrame(y_pred, columns = Y_test.columns)
    for column in Y_test.columns:
        print("Feature:", column)
        print(classification_report(Y_test[column], y_pred_df[column]))


def save_model(model, model_filepath):
    """
    Save the model for classification for model_path
    
    INPUTS:
        model: the model for classification
        model_filepath: the file path for saving the model
    """
    pickle.dump(model, open(model_filepath, "wb"))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()