# FastAPI Railway Template

This is a FastAPI application template ready for deployment on Railway, using Poetry for dependency management.

## Prerequisites

- Python 3.11 or higher
- Poetry (install with `curl -sSL https://install.python-poetry.org | python3 -`)

## Local Development

1. Install dependencies with Poetry:
```bash
poetry install
```

2. Run the application:
```bash
poetry run uvicorn main:app --reload
```

The API will be available at `http://localhost:8000`

## API Documentation

Once the application is running, you can access:
- Interactive API documentation at `/docs`
- Alternative API documentation at `/redoc`

## Deployment on Railway

1. Create a new project on [Railway](https://railway.app/)
2. Connect your GitHub repository
3. Railway will automatically detect the Dockerfile and deploy your application

## Available Endpoints

- `GET /`: Welcome message
- `GET /health`: Health check endpoint

## Environment Variables

No environment variables are required for basic setup. Add any additional variables you need in a `.env` file:

```env
# Example environment variables
# DATABASE_URL=postgresql://user:password@localhost:5432/dbname
```

## Adding Dependencies

To add new dependencies:
```bash
poetry add package-name
```

To add development dependencies:
```bash
poetry add --group dev package-name
``` 