FROM python:3
ADD requirements.txt /
RUN pip install -r requirements.txt
ADD timeToStop.py /
CMD [ "python", "./timeToStop.py" ]
