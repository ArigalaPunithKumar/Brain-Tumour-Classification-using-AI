# Use an official lightweight Python image
FROM python:3.10-slim

# Set up a non-root user (Hugging Face requirement)
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Set the working directory
WORKDIR $HOME/app

# Copy the application code and models into the container
COPY --chown=user:user Frontend/life-care $HOME/app

# Install dependencies (using the CPU version of PyTorch we set earlier)
RUN pip install --no-cache-dir -r requirements.txt

# Start the Flask app using Gunicorn on the dynamic PORT provided by Railway
CMD gunicorn -b 0.0.0.0:$PORT app:app
