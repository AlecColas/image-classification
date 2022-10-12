unittest:
	python3 -m pytest tests

coverage:
	# python3 -m pytest --cov knn.py --cov read_cifar.py --cov-report html tests/
	python3 -m pytest --cov-report html tests/

run: 
	python3 main.py