# The agent sandbox's only way out: squid, a web proxy that lets through Claude's API alone.
# It is the only container on the agent sandbox's network that can also reach the internet.
FROM alpine:3.20
RUN apk add --no-cache squid
COPY squid.conf /etc/squid/squid.conf
# -N: stay in the foreground, as Docker needs. -d 1: print its startup log to `docker logs`,
# which is how loop.py knows the proxy is ready.
CMD ["squid", "-N", "-d", "1"]
