# Base Python image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy only requirements first (optimizes Docker cache)
COPY requirements.txt .

# Install dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy rest of the code after dependencies (caching benefit)
COPY Scripts/ Scripts/

# Create necessary directories
RUN mkdir -p output data

# Set environment variables (optional, but useful)
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Set default command (optional)
CMD ["python", "Scripts/generate_pa_data.py", "--help"]
