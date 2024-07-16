from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import numpy as np
from keras.models import model_from_json
import logging
from werkzeug.utils import secure_filename
import os

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Configuration settings
CONFIG = {
    'UPLOAD_FOLDER': '/home/yfang/dissertation',
    'MODEL_JSON': '/home/yfang/dissertation/model.arch.json',
    'MODEL_WEIGHTS': '/home/yfang/dissertation/model.weights.h5',
    'ALLOWED_EXTENSIONS': {'txt', 'csv'},
}

app.config.update(CONFIG)

# Ensure the upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


# Load model from JSON file
def load_model():
    try:
        with open(app.config['MODEL_JSON'], 'r') as json_file:
            loaded_model_json = json_file.read()
        model = model_from_json(loaded_model_json)
        model.load_weights(app.config['MODEL_WEIGHTS'])
        logger.info("Model loaded successfully")
        return model
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return None


global_model = load_model()


# One-hot encoding of sequences
def one_hot_encode(sequences):
    # horizontal one-hot encoding
    sequence_length = len(sequences[0])
    integer_type = np.int32
    integer_array = LabelEncoder().fit(np.array(('ACGTN',)).view(integer_type)).transform(
        sequences.view(integer_type)).reshape(len(sequences), sequence_length)
    one_hot_encoding = OneHotEncoder(
        sparse_output=False, categories=[[0, 1, 2, 3, 4]] * sequence_length, dtype=integer_type).fit_transform(
        integer_array)

    return one_hot_encoding.reshape(
        len(sequences), 1, sequence_length, 5).swapaxes(2, 3)[:, :, [0, 1, 2, 3], :]


# Pad or trim the sequence to match the required length
def pad_sequence(seq, max_length):
    if len(seq) > max_length:
        seq = seq[-max_length:].upper()
    else:
        seq = 'N' * (max_length - len(seq)) + seq

    return seq


# Process the raw data into sequences suitable for prediction
def process_seqs(data, seq_length=150):
    seqs = []
    for line in data:
        parts = line.split()
        if parts:
            sequence = parts[0].upper()
            filtered_sequence = ''.join([x for x in sequence if x in 'ACGTN'])
            if len(filtered_sequence) >= seq_length:
                seqs.append(filtered_sequence)
            else:
                seqs.append(filtered_sequence.ljust(seq_length, 'N'))
    padded_seqs = np.array([pad_sequence(s, seq_length) for s in seqs])
    X = one_hot_encode(padded_seqs)
    return X


# Check if the file extension is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


# Handle file upload process
def handle_file_upload():
    if 'file' not in request.files:
        return None, {'error': 'No file part'}, 400
    file = request.files['file']
    if file.filename == '' or not allowed_file(file.filename):
        return None, {'error': 'No selected file or invalid file type'}, 400
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    return filepath, None, None


# Process sequences and perform predictions, then return the results
def perform_prediction_and_respond(sequences):
    try:
        processed_sequences = process_seqs(sequences, 150)
        predictions = global_model.predict(processed_sequences.reshape(len(processed_sequences), 4, 150))
        activities = [float(pred) for pred in predictions.flatten()]
        return jsonify({'activities': activities})
    except Exception as e:
        logger.error(f"Error processing sequences: {e}")
        return jsonify({'error': 'Error processing the sequences'}), 500


# Route to upload files and predict DNA sequence activities
@app.route('/upload', methods=['POST'])
def upload_file():
    filepath, error, status = handle_file_upload()
    if error:
        return jsonify(error), status
    try:
        with open(filepath, 'r') as file:
            sequences = file.read().strip().split()
        return perform_prediction_and_respond(sequences)
    except Exception as e:
        logger.error(f"Error processing file: {e}")
        return jsonify({'error': 'Error processing the file'}), 500


# Route to predict DNA sequence activities from provided sequences
@app.route('/predict', methods=['POST'])
def predict_sequences():
    sequences = request.json.get('sequences', [])
    if not sequences:
        return jsonify({'error': 'No DNA sequences provided'}), 400
    return perform_prediction_and_respond(sequences)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
