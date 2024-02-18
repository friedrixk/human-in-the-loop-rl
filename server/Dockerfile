FROM nvidia/cuda:11.2.1-cudnn8-runtime-ubuntu20.04

WORKDIR /app

# Set timezone:
RUN ln -snf /usr/share/zoneinfo/$CONTAINER_TIMEZONE /etc/localtime && echo $CONTAINER_TIMEZONE > /etc/timezone
ARG DEBIAN_FRONTEND=noninteractive

RUN  apt-get update && apt-get install -y --no-install-recommends git gcc python3.10 python3-distutils python3-pip python3-dev && apt-get purge -y --auto-remove && rm -rf /var/lib/apt/lists/*

# Install Python requirements
COPY requirements.txt /tmp/

RUN pip3 install --upgrade pip
RUN pip3 install -r /tmp/requirements.txt
RUN pip3 install --upgrade tensorrt
RUN python3 -m spacy download en_core_web_sm
RUN python3 -c "import nltk;nltk.download('stopwords')"

# Copy the application code
COPY . .
RUN pip install --upgrade -e gym-minigrid
RUN pip install --upgrade -e babyai

# Make sure python output is printed directly to stdout
ENV PYTHONUNBUFFERED=1

# Add to PYTHONPATH
ENV PYTHONPATH=/app

EXPOSE 8050 8050

RUN ls -la
RUN pwd

# Run the application
ENTRYPOINT ["python3", "app.py"]

