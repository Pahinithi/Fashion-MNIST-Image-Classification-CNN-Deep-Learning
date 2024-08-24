# Use a specific Python version with slim variant for smaller image size.
FROM python:3.10-slim

# Set environment variable to ensure the Python output is not buffered
ENV PYTHONUNBUFFERED=1

# Set the working directory inside the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install system dependencies needed for image processing (PIL)
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Expose port 8501 to make Streamlit accessible
EXPOSE 8501

# Create the .streamlit directory and copy configuration files
RUN mkdir -p ~/.streamlit
COPY config.toml ~/.streamlit/config.toml
COPY credentials.toml ~/.streamlit/credentials.toml

# Set the default command to run the Streamlit app, listening on all available interfaces
ENTRYPOINT ["streamlit", "run"]

# Specify the script to run when the container starts
CMD ["main.py", "--server.port=8501", "--server.address=0.0.0.0"]
