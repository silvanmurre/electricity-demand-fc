FROM python:3.11.4-slim

WORKDIR /app

COPY . /app

# Install system dependencies
RUN apt-get update \
    && apt-get install -y libeccodes-dev gcc \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --use-pep517 -e .

CMD ["python", "src/electricity_demand_fc/main.py"]