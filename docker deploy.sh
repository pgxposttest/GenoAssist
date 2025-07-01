sudo systemctl start docker
cd /home/manjaro/PycharmProjects/llmTests/chainlit-gcp
docker build -t chainlit-gcp-image .

if docker run -p 8000:8000 chainlit-gcp-image; then
  echo "Managed to run docker image locally"
  echo "Deploying docker image to google cloud"
  docker tag chainlit-gcp-image:latest asia-southeast1-docker.pkg.dev/cultural-fit-310314/llm/genoassist:latest
  docker push asia-southeast1-docker.pkg.dev/cultural-fit-310314/llm/genoassist:latest
else
  echo "Something went wrong with docker image..."
fi

