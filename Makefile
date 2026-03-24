run_api:
	uvicorn fast_api.app:app --reload --port 8080

docker_build:
	docker build -t venturepulse .

docker_run:
	docker run -p 8080:8080 venturepulse

gcloud_build:
	gcloud builds submit --tag gcr.io/le-wagon-data-science-485511/venturepulse

gcloud_deploy:
	gcloud run deploy venturepulse --image gcr.io/le-wagon-data-science-485511/venturepulse --platform managed --region europe-west1 --allow-unauthenticated
