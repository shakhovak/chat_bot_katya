FROM python:3.7.4

WORKDIR /code

COPY ./requirements.txt /code//requirements.txt

RUN pip install --no-cache-dir -r /code/requirements.txt

COPY . /code

RUN useradd -m -u 1000 user

USER user

ENV HOME=/home/user \
	PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

COPY --chown=user . $HOME/app

CMD ["gunicorn", "-b", "0.0.0.0:7860", "app:app"]