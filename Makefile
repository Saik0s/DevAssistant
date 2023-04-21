
all:
	python -u -m main

test:
	python -u -m main --test

docker:
	docker run -i $$(docker build -q . --build-arg openai_key=$$OPENAI_API_KEY)
