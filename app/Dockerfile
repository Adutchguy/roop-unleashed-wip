FROM python:3.11

# making app folder
WORKDIR /app

# copying files
COPY . .

# installing requirements
RUN apt-get update 
RUN apt-get install ffmpeg -y
RUN pip install --upgrade pip 
RUN pip install -r ./requirements.txt 

# launching gradio app 
ENV GRADIO_SERVER_NAME="0.0.0.0"
EXPOSE 7860
ENTRYPOINT python ./run.py