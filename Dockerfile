# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=utf-8 \
    PORT=8000

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY requirements.lock /app/
# (Fallback to requirements.txt if lock doesn't exist or isn't used by pip directly,
# but assuming requirements.lock is a pip-compatible requirements file since install.ps1 uses `pip install -r requirements.lock`)

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.lock

# Copy the rest of the application code
COPY . /app/

# Expose the port the app runs on (adjust if your app uses a different port)
EXPOSE 8000

# Create necessary directories for local storage and volumes
RUN mkdir -p /app/data /app/memory /app/chroma_db /app/logs /app/outputs /app/workspace /app/runtime

# Healthcheck to ensure the container is running correctly
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# Command to run the application
CMD ["python", "main.py"]
