FROM python:3.9.13

WORKDIR /app

COPY requirements.txt /app

RUN pip install --no-cache-dir -r requirements.txt

COPY . /app

RUN useradd -m -u 1000 user

USER user

ENV HOME=/home/user \
	PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

EXPOSE 5000

COPY --chown=user . $HOME/app

CMD ["gunicorn", "--timeout", "1000", "app:app", "-b", "0.0.0.0:5000"]