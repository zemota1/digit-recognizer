FROM python:3
WORKDIR /code
RUN curl -sSL https://raw.githubusercontent.com/sdispater/poetry/master/get-poetry.py | python
COPY src/read_s3.py /code/
COPY src/config/key.csv /code/config/
COPY ./pyproject.toml /code/
ENV PATH="${PATH}:/root/.poetry/bin"
RUN poetry install
CMD ["/root/.poetry/bin/poetry", "run", "python", "read_s3.py"]