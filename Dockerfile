FROM python:3.10-slim

WORKDIR /app

# Copiamos solo el requirements primero para aprovechar el caché
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiamos todo lo demás
COPY . .

# Exponemos el puerto de FastAPI
EXPOSE 8000

# Lanzamos la app
CMD ["uvicorn", "src.model_deploy:app", "--host", "0.0.0.0", "--port", "8000"]