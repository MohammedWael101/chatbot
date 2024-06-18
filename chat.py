from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import nltk
import numpy as np
import random
import pickle
from time import sleep
from nltk.stem.lancaster import LancasterStemmer
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from dotenv import load_dotenv
import os


load_dotenv()

stemmer = LancasterStemmer()

# Load intents file
with open("intents2.json") as file:
    data = json.load(file)
    
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes    

# Load preprocessed data or process it if not available
try:
    with open("data.pickle", "rb") as f:
        words, labels, training, output = pickle.load(f)
except:
    words = []
    labels = []
    docs_x = []
    docs_y = []

    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            wrds = nltk.word_tokenize(pattern)
            words.extend(wrds)
            docs_x.append(wrds)
            docs_y.append(intent["tag"])

        if intent["tag"] not in labels:
            labels.append(intent["tag"])

    words = [stemmer.stem(w.lower()) for w in words if w != "?"]
    words = sorted(list(set(words)))

    labels = sorted(labels)

    training = []
    output = []

    out_empty = [0 for _ in range(len(labels))]

    for x, doc in enumerate(docs_x):
        bag = []

        wrds = [stemmer.stem(w) for w in doc]

        for w in words:
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)

        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1

        training.append(bag)
        output.append(output_row)

    training = np.array(training)
    output = np.array(output)

    with open("data.pickle", "wb") as f:
        pickle.dump((words, labels, training, output), f)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(training, output, test_size=0.2)

# Define the Keras model
model = Sequential()
model.add(Dense(20, input_shape=(len(X_train[0]),), activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(len(y_train[0]), activation='softmax'))

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.01), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=200, batch_size=8, verbose=1)

# Save the model
model.save("model.h5")

# Evaluate the model
#   y_train_pred = model.predict(X_train)
#   f1_train = f1_score(y_train.argmax(axis=1), y_train_pred.argmax(axis=1), average='weighted')
#   print(f"F1 Score (Training): {f1_train:.4f}")

#   test_loss, test_acc = model.evaluate(X_test, y_test)
#   print(f"Test Loss: {test_loss:.4f}")
#   print(f"Test Accuracy: {test_acc:.4f}")

def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1

    return np.array(bag)

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.form.get('message')
    print(user_input)
    if not user_input:
        return jsonify({"error": "No message provided"}), 400
    
    results = model.predict(np.array([bag_of_words(user_input, words)]))[0]
    results_index = np.argmax(results)
    tag = labels[results_index]
    
    if results[results_index] > 0.8:
        for tg in data["intents"]:
            if tg['tag'] == tag:
                responses = tg['responses']
        sleep(1)
        bot_response = random.choice(responses)
    else:
        bot_response = "I don't understand!"

    return jsonify({"response": bot_response})
if __name__ == '__main__':
    # Run Flask app on port 5000
    port = int(os.getenv('FLASK_RUN_PORT', 5000))
    app.run(debug=True,port=port)
