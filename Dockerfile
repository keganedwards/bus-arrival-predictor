FROM alpine:3.17.0
RUN apk add --no-cache --update \
    python3 python3-dev gcc \
    gfortran musl-dev g++ \
    libffi-dev openssl-dev \
    libxml2 libxml2-dev \
    libxslt libxslt-dev \
    libjpeg-turbo-dev zlib-dev
   
ADD requirements.txt /
RUN python3 -m ensurepip --upgrade
RUN pip3 install -r requirements.txt
ADD timeToStop.py /
CMD [ "python3", "./timeToStop.py" ]
