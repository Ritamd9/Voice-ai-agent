PYTHON ?= python3

.PHONY: install run test

install:
	$(PYTHON) -m pip install -r requirements.txt

run:
	streamlit run app.py

test:
	PYTHONPATH=src $(PYTHON) -m unittest discover -s tests -p "test_*.py"

