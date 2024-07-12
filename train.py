from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, LabelEncoder
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.models import load_model
import pandas as pd
import pickle

# Load the combined dataset
df_combined = pd.read_csv('combined_fallacies_text.csv',  encoding='utf-8')
df_combined.columns = ['fallacy', 'text']

# Encode labels using LabelEncoder
label_encoder = LabelEncoder()
df_combined['class_index'] = label_encoder.fit_transform(df_combined['fallacy'])

# TF-IDF vectorization
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df_combined['text']).toarray()
Y = df_combined['class_index']

# Split data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=22)

# Define the neural network model
model = Sequential()
model.add(Dense(128, input_shape=(X_train.shape[1],), activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(len(label_encoder.classes_), activation='softmax'))  # Output layer for multi-class classification

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, Y_train, epochs=10, batch_size=16, validation_data=(X_test, Y_test))

# Evaluate the model on test data
test_loss, test_accuracy = model.evaluate(X_test, Y_test)
print(f"Test Accuracy: {test_accuracy}")

# Predict using the trained model
predictions = model.predict(X_test)
predicted_labels = label_encoder.inverse_transform(predictions.argmax(axis=-1))
print("Predicted Labels:", predicted_labels)

# Save the model in HDF5 format
with open('combined_fallacy_model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Save the vectorizer to a file using pickle
with open('combined_fallacy_vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)