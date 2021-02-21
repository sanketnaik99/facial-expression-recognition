import sys
import torch
import torchvision.transforms as tt
from models.convnet import ConvNet
from cv2 import CascadeClassifier, COLOR_BGR2GRAY, cvtColor, FONT_HERSHEY_SIMPLEX, putText, rectangle, resize, imread, imwrite
from PIL import Image
import io
import torch.jit as jit


classes = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']


def load_image(imagePath):
    """
    Load the image from the specified folder.
    The imread function from OpenCV is used to read the image.
    """
    print(f'\n\nThe image path is {imagePath}')
    print('Loading the input image...')
    try:
        input_image = imread(imagePath)
        # Check if the input image is valid
        if input_image is None:
            raise IOError
        print('Input Image Loaded Successfully')
    except IOError:
        print('ERROR: Could not Load the input Image')
        sys.exit(1)

    return input_image


def init_face_classifier():
    """
    Load the Cascade Classifier and initialize it with the weights from the Haar Cascade Classifier.
    """
    print('\n\nLoading the Cascade Classifier')
    try:
        facec = CascadeClassifier('models/haarcascade_frontalface_default.xml')
        print('Cascade Classifier Loaded Successfully')
    except:
        print('ERROR: Could not Load the Cascade Classifier')
        sys.exit(1)

    return facec


def init_model():
    """
    Load the saved quantized model for inference.
    """
    # Initialize the Model using the saved Quantized Model

    try:
        torch.backends.quantized.engine = 'qnnpack'
        print('\n\nInitializing Model...')
        model = jit.load(
            'models/convnet-traced-new.pt', map_location='cpu')
        print('Model Loaded Successfully')
    except Exception as e:
        print(e)
        print('ERROR: Could not Initialize the Model.')
        sys.exit(1)

    return model


def predict(image, facec, model):
    """
    This function processes the image, detects the faces on the image and predicts the classes of faces on the image.
    """
    predictions = []
    # Convert the image into a grayscale image
    grayscale_image = cvtColor(image, COLOR_BGR2GRAY)
    # Detect Faces from the grayscale image
    faces = facec.detectMultiScale(grayscale_image, 1.3, 5)
    current = 1
    # Run a loop over all the detected faces
    for (x, y, w, h) in faces:
        # Crop the face into a separate image
        cropped_face = grayscale_image[y:y+h, x:x+w]
        # Resize the image into an image of size 48x48
        resized_face = resize(cropped_face, (48, 48))
        # Convert the image into a PIL image
        PIL_image = Image.fromarray(resized_face)
        # Preprocess the image using torchvision.transforms
        preprocess = tt.Compose([tt.Grayscale(3), tt.ToTensor()])
        processed_image = preprocess(PIL_image)
        print(f'Processed Image Shape => {processed_image.shape}')
        # Run predictions
        batch_t = torch.unsqueeze(processed_image, 0)
        with torch.no_grad():
            out = model(batch_t)
            _, pred = torch.max(out, 1)
        # Get the predicted class
        prediction = classes[pred[0].item()]
        predictions.append(prediction)
        print(f'\nPrediction for Face {current} => {prediction}')
        # Add the rectangle and label to the original image
        putText(image, prediction, (x, y-10),
                FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

        current += 1
    imwrite('result.jpg', image)
    return predictions


if __name__ == "__main__":
    imagePath = sys.argv[1]
    image = load_image(imagePath)
    facec = init_face_classifier()
    model = init_model()
    # This function will return a string array contianing all the predictions
    predictions = predict(image, facec, model)
    print(f'Predictions => {predictions}')
