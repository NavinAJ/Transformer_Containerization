# Containerizing Transformer 
![Containerizing Transformer](https://text.relipasoft.com/wp-content/uploads/2017/12/download.png "Docker")
A Sample project for Containerizing Transformer model (Bert Sentiment analysis).
- The Model is trained on GPU but the docker image generated supports GPU/CPU runtime.
- The Docker image generated is available in the dockerhub for reference.
```sh
docker pull navinprasad/sa_transformer_containerization:latest
```
- Once the image is loaded or try running the docker image.
```sh
docker container run -d -p 80:5000 --name bert navinprasad/sa_transformer_containerization:latest
```

##### Prerequisite
- Install docker
