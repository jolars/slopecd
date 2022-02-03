all: install

install:
	pip install -e code

test:
	python -m unittest discover code/tests


