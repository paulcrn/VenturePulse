train:
	python -c "from project_logic.predict import train_model; from project_logic.registry import save_model; save_model(train_model())"

run_api:
	uvicorn fast_api.app:app --reload --port 8080

docker_build:
	docker build -t venturepulse .

docker_run:
	docker run -p 8080:8080 venturepulse
