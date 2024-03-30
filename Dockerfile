FROM python:3.10

WORKDIR /app

# Copying requirements and installing dependencies
COPY requirements.txt .
# Install system dependencies including GLib for OpenCV
RUN apt-get update && \
    apt-get install -y libgl1-mesa-glx libglib2.0-0
RUN pip install gunicorn
RUN pip install -r requirements.txt

# Copying the Flask app files
COPY app.py .
COPY ./templates ./templates
COPY ./static ./static

# Adding the pre-trained model file
ADD oily_model_final8.h5 .

# Exposing port 5000 from the container
EXPOSE 5000

# Starting the Flask application using Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]
