# SMS Spam Classifier

An end-to-end machine learning project that classifies SMS messages as spam or ham.

The project includes:

- TF-IDF text features
- Character-level TF-IDF features
- Engineered message features
- Logistic Regression
- FastAPI inference API
- Automated API tests with pytest
- Docker containerization

## Model Performance

The final model was evaluated on a held-out test set of 1,034 SMS messages.

| Metric | Ham | Spam |
|---|---:|---:|
| Precision | 0.99 | 1.00 |
| Recall | 1.00 | 0.94 |
| F1-score | 1.00 | 0.97 |

Overall accuracy: **99.23%**

Confusion matrix:

```text
              Predicted
              Ham  Spam
Actual Ham    903    0
Actual Spam     8  123
```

The model uses a combination of word-level TF-IDF, character-level TF-IDF, and engineered message features with a balanced Logistic Regression classifier.

## Running the API Locally

Install the development dependencies:

```bash
pip install -r requirements-dev.txt
```

Start the API:

```bash
uvicorn app.main:app --reload
```

The API will be available at:

```text
http://127.0.0.1:8000
```

Interactive API documentation:

```text
http://127.0.0.1:8000/docs
```

### Health Check

```http
GET /health
```

Example response:

```json
{
  "status": "healthy"
}
```

### Prediction

```http
POST /predict
```

Example request:

```json
{
  "message": "Congratulations! You have won a free prize!"
}
```

Example response:

```json
{
  "prediction": "spam",
  "spam_probability": 0.8878
}
```

## Running with Docker

Build the Docker image:

```bash
docker build -t sms-spam-classifier .
```

Start the container:

```bash
docker run -d -p 8000:8000 --name sms-spam-api sms-spam-classifier
```

Check the container:

```bash
docker ps
```

The container should eventually show:

```text
healthy
```

The API is then available at:

```text
http://localhost:8000
```

Interactive documentation:

```text
http://localhost:8000/docs
```

Stop the container:

```bash
docker stop sms-spam-api
```

Start the existing container again:

```bash
docker start sms-spam-api
```

## Running Tests

Run the automated API tests with:

```bash
python -m pytest
```

The test suite currently covers:

- Health endpoint
- Spam prediction
- Ham prediction
- Empty-message validation

All tests use FastAPI's `TestClient` and pytest fixtures.

## Project Structure

```text
sms-spam-classifier/
├── app/
│   ├── __init__.py
│   └── main.py
├── data/
│   └── raw/
│       └── SMSSpamCollection
├── models/
│   ├── feature_scaler.pkl
│   ├── final_spam_pipeline.pkl
│   ├── spam_model.pkl
│   └── tfidf_vectorizer.pkl
├── notebooks/
│   └── exploration.ipynb
├── src/
│   ├── predict.py
│   ├── preprocessing.py
│   └── train.py
├── tests/
│   ├── conftest.py
│   └── test_api.py
├── .dockerignore
├── .gitignore
├── Dockerfile
├── requirements.txt
├── requirements-dev.txt
└── readme.md
```

## Development Workflow

1. Install development dependencies:

   ```bash
   pip install -r requirements-dev.txt
   ```

2. Run the tests:

   ```bash
   python -m pytest
   ```

3. Start the API locally:

   ```bash
   uvicorn app.main:app --reload
   ```

4. Build the Docker image:

   ```bash
   docker build -t sms-spam-classifier .
   ```

5. Run the Docker container:

   ```bash
   docker run -d -p 8000:8000 --name sms-spam-api sms-spam-classifier
   ```

6. Verify the container health:

   ```bash
   docker ps
   ```