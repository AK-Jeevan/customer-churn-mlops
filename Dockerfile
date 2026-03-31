# Base image (Python)
FROM python:3.10

# Set working directory inside container
WORKDIR /app

# Copy all project files into container
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port (FastAPI runs on 8000)
EXPOSE 8000

# Command to run API
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]