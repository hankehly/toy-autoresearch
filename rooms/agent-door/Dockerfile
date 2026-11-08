# The agent room's door: a web proxy (squid) that lets through Claude's API and nothing else.
# It is the only container on the agent room's network that can also reach the internet.
FROM alpine:3.20
RUN apk add --no-cache squid
COPY squid.conf /etc/squid/squid.conf
# -N: stay in the foreground, as Docker needs. -d 1: print startup notes to `docker logs`,
# which is how loop.py knows the door is open.
CMD ["squid", "-N", "-d", "1"]
