# Facial Expression Recognition

A Facial Expression Recognition system developed using PyTorch and OpenCV.

## Instructions

The project requires the following dependencies to be installed. Additional dependencies may be required so it is recommended to install all the required dependencies from the `requirements.txt` file.

Required Dependencies -

- torch==1.6.0+cpu
- torchvision==0.7.0+cpu
- opencv-python==4.5.1.4

### Clone the project and initialize a virtual environment

```bash
# Clone the repository
git clone https://github.com/sanketnaik99/facial-expression-recognition.git

# cd into the directory
cd facial-expression-recognition

# Initialize the virtual environment
python3 -m venv env
```

### Activate the environment and install the dependencies

```bash
# Activate the virtual environment
source env/bin/activate

# Install the dependencies from requirements.txt
pip install -r requirements.txt
```

### Run the demo code to check if everything is working

```bash
# Run the Demo Code
python fer_demo.py images/tony-stark.jpg
```

### Result

You will find the result image stored as `result.jpg` in the root directory. Verify that it has been labelled with the correct expression and has detected the face/faces properly.
