from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
import json
import pickle

app = Flask(__name__)

# Load the intents JSON file to use in response selection
with open("intents.json") as file:
    data = json.load(file)

# Load the trained chatbot model
model = tf.keras.models.load_model('chat_model')

# Load tokenizer and label encoder objects
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

with open('label_encoder.pickle', 'rb') as enc:
    lbl_encoder = pickle.load(enc)

# Define the maximum length of input sequences
max_len = 20

@app.route('/chat', methods=['POST'])
def chat():
    try:
        # Ensure there is a message to process
        message = request.json.get('message', '')
        if not message:
            return jsonify({'error': 'Message is required'}), 400

        # Preprocess the input message
        sequence = tokenizer.texts_to_sequences([message])
        padded_sequence = tf.keras.preprocessing.sequence.pad_sequences(sequence, truncating='post', maxlen=max_len)

        # Predict the tag for the input message
        result = model.predict(padded_sequence)
        tag = lbl_encoder.inverse_transform([np.argmax(result)])

        # Find a response corresponding to the predicted tag
        response = {"tag": str(tag[0])}
        for i in data['intents']:
            if i['tag'] == tag:
                response["response"] = np.random.choice(i['responses'])
                break

        return jsonify(response)
    except Exception as e:
        # Handle any errors that occur during processing
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
