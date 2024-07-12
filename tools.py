import numpy as np
import pandas as pd
from keras.models import load_model
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
import pickle
from sklearn.preprocessing import LabelEncoder
from crewai import Agent, Task
from langchain.tools import tool
import json
import subprocess


class Scraper():

 @tool("scrape google scholar for evidence to refute arguments and improve them")
 def scrape_data(negation_query: str):
    """Useful to use to find evidence from established sources with detail and direct citations from google scholar.
     This is to help make your counterargument much stronger against the given evidence as well as for making a better argument. """

    try:
    # Runs Node.js scraper for Google Scholar scraping
      result = subprocess.run(['node', 'scraper.js', negation_query], capture_output=True, text=True, encoding='utf-8')
      print("Captured Node.js script output")
      scraped_text = result.stdout
    
    # Limit the scraped_text to a maximum of 750 words
      words = scraped_text.split()
      if len(words) > 750:
         scraped_text = ' '.join(words[:750])
    except subprocess.CalledProcessError as e:
      print(f"Error executing Node.js script: {e}")

    return {"evidence": scraped_text}
 
class Fallacies():

 @tool("Use as a tool to structure your refutations by copying the structure of the fallacies")
 def find_fallacies(query: str, argument: str, evidence: str):
    """Useful to find all possible refutations(dont find logical fallacies) from evidence 
    given an example that you must emulate,
    for each item you add, you can choose to only use 'refutation' if any of  them exist in the example,
      Give an appropiate amount of items. DO not split the sentences like the example below. You must choose to use 2 or more sentences with context and use a fallacy and/or refutation that captures this context and reasoning. """

    #example:
    text = "In addressing climate change, it's important to acknowledge both natu)ral climatic cycles and the significant impact of human activities. Extensive scientific research consistently shows that human actions, particularly greenhouse gas emissions, significantly contribute to the observed global warming trends. This body of research has undergone rigorous scrutiny through independent verification and peer review processes. Efforts to mitigate climate change are critical for protecting ecosystems, fostering sustainable practices, and preparing for future challenges. Collaboration among scientists, policymakers, and communities is essential for developing effective solutions to tackle this global issue."
    sentences = text.strip().split('. ')
    label_encoder = LabelEncoder()
   
    df_combined = pd.read_csv('combined_fallacies_text.csv',  encoding='utf-8')
    df_combined['class_index'] = label_encoder.fit_transform(df_combined['fallacy'])
    unique_classes = df_combined['fallacy'].unique()
      #load model
    model_path = 'combined_fallacy_model.h5'
    try:
       model_loaded = keras.models.load_model(model_path)
       print("Model loaded successfully.")
    except Exception as e:
       print(f"Error loading model: {e}")
         # Load the vectorizer
    with open('combined_fallacy_vectorizer.pkl', 'rb') as f:
       vectorizer = pickle.load(f)


    json_array = []


    for i, sentence in enumerate(sentences):
       X_sentence = vectorizer.transform([sentence]).toarray()
       predictions = model_loaded.predict(X_sentence)

   
       predicted_class_index = np.argmax(predictions, axis=-1)[0]
       predicted_class = label_encoder.inverse_transform([predicted_class_index])[0]

       probabilities = predictions[0]
       probabilities_decimal = ["{:.6f}".format(p) for p in probabilities]


       sentence_json = {
                'sentence': sentence,
                'refutation':'',#add your refutation here
                'prediction': predicted_class,  # 
       }
      # Check if every TF-IDF value is less than 0.5
       if all(float(value) < 0.5 for value in probabilities_decimal):
          sentence_json['prediction'] = 'None'

       json_array.append(sentence_json)
      
    print(json.dumps(json_array, indent=2))
    return {"fallacies":json_array, "possible_classes": unique_classes}
      

   

  
#perhaps you can find an ideal way of structuring an argument here if as so:
