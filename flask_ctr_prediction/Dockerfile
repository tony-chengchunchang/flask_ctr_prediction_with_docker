FROM tensorflow/tensorflow:2.5.0-gpu-jupyter
EXPOSE 5000
ENV FLASK_APP=app.py
WORKDIR /usr/src/app
COPY . .
RUN pip install -r requirements.txt
CMD ["flask", "run", "-h", "0.0.0.0"]
