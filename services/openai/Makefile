APP_NAME=fastapi-openai
PORT=8000

# Build the Docker image
docker-build:
	docker build -t $(APP_NAME) .

# Run the container (production mode)
docker-run:
	docker run -p $(PORT):8000 --env-file .env $(APP_NAME)

# Run with auto-reload (development mode)
docker-dev:
	docker run -it --rm \
		-p $(PORT):8000 \
		--env-file .env \
		-v $(PWD):/app \
		$(APP_NAME) \
		uvicorn main:app --reload --host 0.0.0.0 --port $(PORT)

# Stop all running containers of this image
docker-stop:
	docker ps -q --filter ancestor=$(APP_NAME) | xargs -r docker stop

# Remove built image
docker-clean:
	docker rmi $(APP_NAME) || true

dev:
	uvicorn main:app --reload --port 8000

install:
	python3 -m pip install -r requirements.txt

lint:
	python3 -m ruff check .

format:
	python3 -m ruff format .
