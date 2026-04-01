.PHONY: install pipeline dashboard test clean

install:
	pip install -r requirements.txt

pipeline:
	python run.py

pipeline-fast:
	python run.py --skip-data

dashboard:
	streamlit run src/dashboard/app.py

test:
	pytest tests/ -v --tb=short

test-cov:
	pytest tests/ -v --tb=short --cov=src --cov-report=term-missing

clean:
	find . -type d -name __pycache__ | xargs rm -rf
	find . -name "*.pyc" | xargs rm -f
	rm -f data/revenue_intelligence.duckdb
	rm -f data/mart/churn_model.pkl

clean-data:
	rm -rf data/raw/*.parquet data/raw/*.csv
	rm -rf data/staging/*.parquet
	rm -rf data/intermediate/*.parquet
	rm -rf data/mart/*.parquet
	rm -f data/revenue_intelligence.duckdb
