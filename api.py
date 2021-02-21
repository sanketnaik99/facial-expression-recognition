from flask import Flask, jsonify, request
from fer_demo import load_image, init_model, init_face_classifier, predict

app = Flask(__name__)


@app.route('/')
def index():
    return """
    <h1>Welcome to FER</h1>
    <p>This is a simple flask app that serves the FER Model.</p>
    """


@app.route('/predict', methods=['POST'])
def predict_image():
    image = request.files['image']
    imagePath = './input-image.jpg'
    image.save(imagePath)
    image = load_image(imagePath)
    facec = init_face_classifier()
    model = init_model()
    # This function will return a string array contianing all the predictions
    predictions = predict(image, facec, model)
    print(f'Predictions => {predictions}')
    return jsonify(expression=predictions), 200


if __name__ == "__main__":
    app.run(debug=True)
