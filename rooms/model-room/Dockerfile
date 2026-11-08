# The model room: Python with numpy and pandas (the same versions as on the Mac), plus
# the runner. judge.py starts it with the locks on and hands it model.py, read-only.
FROM python:3.12-slim
RUN pip install --no-cache-dir numpy==2.2.3 pandas==2.2.3
COPY runner.py /room/runner.py
# A model that runs for 10 minutes is stopped, and the judge reports a crash.
ENTRYPOINT ["timeout", "600", "python", "/room/runner.py"]
