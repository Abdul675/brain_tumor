FROM python:3.13.6-slim

WORKDIR /app

# Install system dependencies (needed for opencv)
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8502

CMD ["python","-m","streamlit", "run", "main.py", "--server.port", "8502", "--server.address", "0.0.0.0"]
