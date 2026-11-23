# The agent sandbox: Claude Code, plus Python with numpy, pandas and pyarrow in the same
# versions as the model sandbox, so check.py runs model.py the way the judge will. loop.py
# starts it with only the turn's workspace and the agent's months of the data store mounted,
# on a network whose only way out is the proxy.
FROM python:3.12-slim
RUN pip install --no-cache-dir numpy==2.2.3 pandas==2.2.3 pyarrow==26.0.1
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*
RUN useradd --create-home agent
USER agent
# pipefail: if the download fails, the build stops. Without it, bash would run an empty
# script, report success, and leave an image with no Claude Code in it.
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
# The same Claude Code version as on the Mac.
RUN curl -fsSL https://claude.ai/install.sh | bash -s 2.1.319
ENV PATH=/home/agent/.local/bin:$PATH
WORKDIR /work
