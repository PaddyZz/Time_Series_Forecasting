
FROM python:3.10.3
RUN pip install virtualenv
RUN virtualenv /env
ENV VIRTUAL_ENV=/env
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
WORKDIR /app
COPY . /app
RUN python -m pip install --no-cache-dir -r requirements.txt
CMD ["python", "main.py"]
