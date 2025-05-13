FROM python:3.9-slim

WORKDIR /app

COPY Scripts/ Scripts/
COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install -r requirements.txt

RUN mkdir -p output data

# Optional default command
CMD ["python", "Scripts/generate_pa_data.py", "--help"]
