FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

#install dependencies
RUN pip install --no-cache-dir -r requirements.txt

#copy source codes
COPY src/ ./src/
COPY models/ ./models/
COPY data/ ./data/
COPY main.py .

#create a non-root user and activate it
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

#expose port (for later)
EXPOSE 8000

#default command
CMD ["python","main.py","--prompt","To be or not to be","--length","100"]
