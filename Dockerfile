FROM python:3.12-slim
WORKDIR /opt/recovery-lab
COPY . /opt/recovery-lab
RUN python -m pip install --no-cache-dir .
CMD ["python", "-m", "jev_recovery", "benchmark", "--config", "/opt/recovery-lab/configs/research.json", "--output", "/opt/recovery-lab/artifacts/research"]
