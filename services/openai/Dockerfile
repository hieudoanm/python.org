# Use official Python base image
FROM python:3.13-slim-bullseye

# Set working directory inside container
WORKDIR /app

# Upgrade system packages to address vulnerabilities
RUN apt-get update && apt-get upgrade -y && apt-get clean

# Copy dependency files first (for caching)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your app
COPY . .

# Expose FastAPI port
EXPOSE 8000

# Run FastAPI with uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
