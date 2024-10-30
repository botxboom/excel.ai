from flask import Flask, jsonify, request
import os
from excel import upload_excel, process_question

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
DB_FOLDER = 'db'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['DB_FOLDER'] = DB_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(DB_FOLDER, exist_ok=True)

@app.route('/')
def home():
    return "Hello, Flask!"

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # Save the file
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)
    message = upload_excel(file_path)
    
    return jsonify({"message": message, "file_path": file_path}), 200

@app.route('/ask', methods=['POST'])
def ask():
    question = request.form.get('question')

    if not question:
        return jsonify({"error": "No question provided"}), 400

    result = process_question(question)
    
    return jsonify({"response": result}), 200

if __name__ == '__main__':
    app.run(debug=True)
