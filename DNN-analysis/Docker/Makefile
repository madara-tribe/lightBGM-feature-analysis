run:
	docker build -t deeplabv3 .

in:
	nvidia-docker run --runtime=nvidia -v /home/hagi/:/home/app/sig --rm -it deeplabv3:latest bash
