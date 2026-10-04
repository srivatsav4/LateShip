PY ?= python
SOURCE ?= demo   # demo (synthetic) | olist (real data in data/raw/)

.PHONY: install test lint demo-data features train evaluate demo mlflow api dashboard docker-build docker-run

install:
	$(PY) -m pip install -r requirements-dev.txt

test:
	$(PY) -m pytest

lint:
	$(PY) -m ruff check .

demo-data:
	$(PY) scripts/generate_sample_data.py

features: $(if $(filter demo,$(SOURCE)),demo-data)
	$(PY) scripts/build_features.py --source $(SOURCE)

train:
	$(PY) scripts/train_model.py --source $(SOURCE)

evaluate:
	$(PY) scripts/evaluate_model.py --source $(SOURCE)

demo:
	$(PY) scripts/run_demo.py

mlflow:
	$(PY) -m mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000

api:
	$(PY) -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000

dashboard:
	$(PY) -m streamlit run app/dashboard.py

docker-build:
	docker build -t lateship:latest .

docker-run:
	docker run --rm -p 8000:8000 lateship:latest
