
all:
	python -u -m main

test:
	python -u -m main --test --verbose

install:
	pip install -r requirements.txt
	cd modules/tools && npm install

docker:
	docker build -t dev_assistant .
	docker run -i dev_assistant
