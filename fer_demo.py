import sys
import torch
import torchvision.transforms as tt
from models.convnet import ConvNet
from cv2 import CascadeClassifier, COLOR_BGR2GRAY, cvtColor, FONT_HERSHEY_SIMPLEX, putText, rectangle, resize, imread, imwrite
from PIL import Image

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
        print('Input Image Loaded Successfully')
    except:
        print('ERROR: Could not Load the input Image')
    
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
    
    return facec

def init_model():
    """
    Load the saved quantized model for inference.
    """
    # Initialize the Model using the saved Quantized Model
    print('\n\nInitializing Model...')
    try:
        model = torch.load('models/convnet-quantized-full.pt')
        print('Model Loaded Successfully')
    except:
        print('ERROR: Could not Initialize the Model.')

    return model

def predict(image, facec, model):
    """
    This function processes the image, detects the faces on the image and predicts the classes of faces on the image.
    """
    grayscale_image = cvtColor(image, COLOR_BGR2GRAY)
    faces = facec.detectMultiScale(grayscale_image, 1.3, 5)
    current = 1
    for (x, y, w, h) in faces:
        cropped_face = grayscale_image[y:y+h, x:x+w]
        resized_face = resize(cropped_face, (48,48))
        PIL_image = Image.fromarray(resized_face)
        preprocess = tt.Compose([tt.Grayscale(1), tt.ToTensor()])
        processed_image = preprocess(PIL_image)
        print(f'Processed Image Shape => {processed_image.shape}')
        batch_t = torch.unsqueeze(processed_image, 0) # this will convert the
        with torch.no_grad():
            out = model(batch_t)
            _, pred = torch.max(out, 1)
        prediction = classes[pred[0].item()]
        print(f'\nPrediction for Face {current} => {prediction}')
        putText(image, prediction, (x, y-10), FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        rectangle(image, (x,y), (x+w, y+h), (0, 255, 0), 2)
        current += 1
    imwrite('result.jpg', image)
    return


if __name__ == "__main__":
    imagePath = sys.argv[1]
    image = load_image(imagePath)
    facec = init_face_classifier()
    model = init_model()
    predict(image, facec, model)







