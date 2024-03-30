from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import openai

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
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


@app.route('/')
def index():
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

    # Preprocess the uploaded image
    img = preprocess_image(filename)

    # Make predictions
    predictions = model.predict(img)

    # Interpret predictions
    class_names = ['non-oily', 'oily']
    predicted_class = np.argmax(predictions)
    predicted_label = class_names[predicted_class]
    confidence_level = predictions[0][predicted_class] * 100  # Confidence as percentage

    # Determine OILYNESS level
    oiliness_level = None
    if predicted_label == 'oily':
        if confidence_level > 80:
            oiliness_level = "HIGH-OILYNESS"
        elif confidence_level <= 80 and confidence_level > 60:
            oiliness_level = "MID-OILYNESS"
        else:
            oiliness_level = "LOW-OILYNESS"

    elif predicted_label == 'non-oily':
        oiliness_level = "LOW-OILYNESS"

    # Render the result template with the prediction details
    return render_template('index.html', filename=file.filename, confidence_level=confidence_level,
                           oiliness_level=oiliness_level)


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
    app.run(host='0.0.0.0')
