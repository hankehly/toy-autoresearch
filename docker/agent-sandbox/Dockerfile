# The agent sandbox: Claude Code, plus Python with numpy and pandas in the same versions as the
# model sandbox, so check.py runs model.py the way the judge will. loop.py starts it with only
# the turn's workspace mounted, on a network whose only way out is the proxy.
FROM python:3.12-slim
RUN pip install --no-cache-dir numpy==2.2.3 pandas==2.2.3
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*
RUN useradd --create-home agent
USER agent
# The same Claude Code version as on the Mac.
RUN curl -fsSL https://claude.ai/install.sh | bash -s 2.1.319
ENV PATH=/home/agent/.local/bin:$PATH
WORKDIR /work
