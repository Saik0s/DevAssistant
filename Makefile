
all:
	python -u -m main

docker:
	docker run -i $$(docker build -q . --build-arg openai_key=$$OPENAI_API_KEY)
