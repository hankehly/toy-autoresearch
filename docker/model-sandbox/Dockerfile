# The model sandbox: Python with numpy, pandas and pyarrow (the same versions as on the Mac),
# plus the runner. judge.py starts it with the locks on and hands it model.py and the month
# folders before the first scored month, all read-only.
FROM python:3.12-slim
RUN pip install --no-cache-dir numpy==2.2.3 pandas==2.2.3 pyarrow==26.0.1
COPY runner.py /sandbox/runner.py
# A model that runs for 10 minutes is stopped, and the judge reports a crash.
ENTRYPOINT ["timeout", "600", "python", "/sandbox/runner.py"]
