FROM python:3
WORKDIR /code
RUN curl -sSL https://raw.githubusercontent.com/sdispater/poetry/master/get-poetry.py | python
COPY . /code
ENV PATH="${PATH}:/root/.poetry/bin"
RUN poetry install
CMD ["/root/.poetry/bin/poetry", "run", "python", "src/digit_recognizer.py"]