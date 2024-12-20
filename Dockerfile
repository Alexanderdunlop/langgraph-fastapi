FROM python:3.11-slim

WORKDIR /app

# Install poetry
RUN pip install poetry

# Configure poetry to not create a virtual environment
RUN poetry config virtualenvs.create false

# Copy the entire project first
COPY . .

# Install dependencies
RUN poetry install --no-interaction --no-ansi

# Run the application
CMD ["poetry", "run", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]