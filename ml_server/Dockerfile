FROM ubuntu:20.04

RUN apt-get update -y
RUN apt-get upgrade -y

RUN apt-get install program, program2
RUN pip3 install library, library2


WORKDIR /opt/app

COPY . .
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8080

CMD ["python", "./api.py"]