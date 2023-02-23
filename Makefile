run: 
	python3 main.py

unittest:
	python3 -m pytest tests

coverage:
	python3 -m pytest --cov-report html --cov=modules tests/
