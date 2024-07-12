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
import json


text = "In addressing climate change, it's important to acknowledge both natural climatic cycles and the significant impact of human activities. Extensive scientific research consistently shows that human actions, particularly greenhouse gas emissions, significantly contribute to the observed global warming trends. This body of research has undergone rigorous scrutiny through independent verification and peer review processes. Efforts to mitigate climate change are critical for protecting ecosystems, fostering sustainable practices, and preparing for future challenges. Collaboration among scientists, policymakers, and communities is essential for developing effective solutions to tackle this global issue."
sentences = text.strip().split('. ')
label_encoder = LabelEncoder()

df_combined = pd.read_csv('combined_fallacies_text.csv',  encoding='utf-8')
df_combined['class_index'] = label_encoder.fit_transform(df_combined['fallacy'])

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
unique_classes = df_combined['fallacy'].unique()
print(json.dumps(json_array, indent=2))
print({"fallacies":json_array, "possible_classes": unique_classes})
