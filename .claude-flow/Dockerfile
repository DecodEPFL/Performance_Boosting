FROM python:3.12-slim

WORKDIR /workspace

COPY requirements.txt .
RUN python -m pip install --upgrade pip \
    && python -m pip install --root-user-action=ignore -r requirements.txt

COPY . .

CMD ["python", "nav_experiment.py"]