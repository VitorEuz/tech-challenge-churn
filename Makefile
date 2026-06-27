run:
	python -m uvicorn src.api.main:app --reload

test:
	python -m pytest

lint:
	ruff check .

train:
	python -m src.models.train_model

baseline:
	python -m src.models.baseline_model

predict:
	python -m src.models.predict_model

mlflow:
	mlflow ui