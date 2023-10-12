import string        # used for preprocessing
import re            # used for preprocessing
import nltk          # the Natural Language Toolkit, used for preprocessing
import numpy as np   # used for managing NaNs
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords       # used for preprocessing
from nltk.stem import WordNetLemmatizer # used for preprocessing
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

# Function to lemmatize Words
lemmatizer = WordNetLemmatizer()
def lemmatize(text):
    text = [lemmatizer.lemmatize(token) for token in text]
    return text

#function for data preprocessing of texts
def pre_processing(text):

  text = text.lower()
  text = ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",str(text)).split()) #remove urls
  text = re.sub(r'\d+', '', str(text)) #emove no.s
  text=text.replace('\n',' ')
  text = word_tokenize(text)
  text = [char for char in text if char not in string.punctuation]
  text = [word for word in text if word not in stopwords.words('english')]
  text = lemmatize(text)
  text = ' '.join(text)
  return text


#Applying preprocessing and removing '\n' character

def output_text(df,column_name): 
    for i in range(df.shape[0]):
        df[column_name][i] = pre_processing(str(df[column_name][i])) 

    x = [word_tokenize(word) for word in df[column_name]]   #Tokenizing data for training purpose
    return x


#Preprocessing input, because input should be in same form as training data set
def preprocessing_input(query):
    query = pre_processing(query)
    query = query.replace('\n',' ')         
    return query  
