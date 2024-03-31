# Use a lightweight Python image as a base
FROM amd64/python:3.10.13-slim

# Set the working directory in the container
WORKDIR /app


# Install system dependencies for OpenCV and Flask
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements file to install dependencies
COPY requirements.txt .

# Install Python dependencies
RUN pip install -r requirements.txt

# Install gunicorn
RUN pip install gunicorn

# Copy the rest of the application code
COPY app.py .
COPY ./templates ./templates
COPY ./static ./static
COPY oily_model_final8.h5 .

# Expose the port the app runs on
EXPOSE 5000

# Command to run the application
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]
