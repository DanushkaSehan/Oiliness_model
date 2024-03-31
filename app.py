from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import cv2
import tempfile
import webbrowser  # Import the webbrowser module
import openai

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the trained model
model_path = "oily_model_final8.h5"
model = load_model(model_path)

# Function to preprocess the uploaded image
def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0  # Normalize the image
    return img

# Function to extract skin from the image
def extract_skin(image_path):
    img = cv2.imread(image_path)
    # Convert BGR to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Define range of skin color in HSV
    lower_skin = np.array([0, 48, 80], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)
    # Threshold the HSV image to get only skin colors
    mask = cv2.inRange(hsv, lower_skin, upper_skin)
    # Bitwise-AND mask and original image
    skin = cv2.bitwise_and(img, img, mask=mask)

    # Save the extracted skin image temporarily
    temp_dir = 'static/uploads'
    temp_path = os.path.join(temp_dir, 'skin_' + os.path.basename(image_path))
    cv2.imwrite(temp_path, skin)

    return temp_path


# functions for real-time detection 
# Function to preprocess the frame for prediction
def preprocess_frame(frame):
    frame_resized = cv2.resize(frame, (224, 224))  # Resize frame to match model's expected input size
    frame_normalized = frame_resized / 255.0  # Normalize pixel values
    frame_batch = np.expand_dims(frame_normalized, axis=0)  # Add batch dimension
    return frame_batch

# Function to remove background
def remove_background(frame):
    # Create the background subtractor object
    # Feel free to adjust the history as needed
    backSub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)
    
    # Apply the background subtractor
    fgMask = backSub.apply(frame)
    
    # Get the foreground
    foreground = cv2.bitwise_and(frame, frame, mask=fgMask)
    
    return foreground

# End the Preprocessing and Removal Background part

@app.route('/')
def index():
    return render_template('web.html')

@app.route('/index')
def web():
    return render_template('index.html')

@app.route('/webcam', methods=['GET', 'POST'])
def webcam():
    # Start video capture
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Could not open video device")
        exit()

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        
        if not ret:
            print("Failed to grab frame")
            break
        
        # Remove the background
        frame_no_bg = remove_background(frame)
        
        # Preprocess the frame
        preprocessed_frame = preprocess_frame(frame_no_bg)
        
        # Predict on the preprocessed frame
        predictions = model.predict(preprocessed_frame)
        predicted_class = np.argmax(predictions)
        predicted_label = ['non-oily', 'oily'][predicted_class]
        confidence_level = predictions[0][predicted_class] * 100  # Confidence as a percentage

        # Determine OILYNESS level
        oiliness_level = "LOW-OILYNESS"
        if predicted_label == 'oily':
            if confidence_level > 89:
                oiliness_level = "HIGH-OILYNESS"
            elif confidence_level <= 89 and confidence_level > 70:
                oiliness_level = "MID-OILYNESS"
        
        # Add the prediction result to the frame
        cv2.putText(frame_no_bg, f'OILYNESS Level: {oiliness_level} ({confidence_level:.2f}%)', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        
        # Display the resulting frame
        cv2.imshow('Webcam - Oily Skin Detection', frame_no_bg)
        
        # Press 'q' to exit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything is done, release the capture
    cap.release()
    cv2.destroyAllWindows()
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template('index.html', error='No file part')

    file = request.files['file']
    if file.filename == '':
        return render_template('index.html', error='No selected file')

    filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filename)

    # Extract skin from the uploaded image
    skin_image = extract_skin(filename)

    # Preprocess the extracted skin image
    skin_img_processed = preprocess_image(skin_image)

    # Make predictions
    predictions = model.predict(skin_img_processed)

    # Interpret predictions
    class_names = ['non-oily', 'oily']
    predicted_class = np.argmax(predictions)
    predicted_label = class_names[predicted_class]
    confidence_level = predictions[0][predicted_class] * 100  # Confidence as a percentage

    # Determine OILYNESS level
    oiliness_level = None
    if predicted_label == 'oily':
        if confidence_level > 89:
            oiliness_level = "HIGH-OILYNESS"
        elif confidence_level <= 89 and confidence_level > 70:
            oiliness_level = "MID-OILYNESS"
        else:
            oiliness_level = "LOW-OILYNESS"  
            
    elif predicted_label == 'non-oily':
        oiliness_level = "LOW-OILYNESS"
        
        
    # Render the result template with the prediction details
    return render_template('index.html', filename=file.filename, confidence_level=confidence_level, oiliness_level=oiliness_level, skin_filename=os.path.basename(skin_image))


# Chat BOT-------------------------------------
# Set your OpenAI API key here
openai.api_key = "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"


@app.route("/chatbot", methods=["GET", "POST"])
def chatbot():
    if request.method == "POST":
        user_input = request.form.get("user_input")
        try:
            chat_history = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful skin specialist medical chatbot."},
                    {"role": "user", "content": user_input}
                ]
            )

            chat_response = chat_history["choices"][0]["message"]["content"]
        except Exception as e:
            chat_response = f"Error occurred: {str(e)}"

        return render_template("bot.html", user_input=user_input, chat_response=chat_response)

    return render_template("bot.html")



if __name__ == '__main__':
    app.run(debug=True)
