#!/usr/bin/env python3
import subprocess
import re
import lttngust.loghandler
import logging

logger = logging.getLogger("otel.spans")
lttng_handler = lttngust.loghandler._Handler()
logger.addHandler(lttng_handler)
logger.setLevel(logging.DEBUG)

pattern = re.compile(
    r"\[otel\.javaagent.*?\] INFO.*?LoggingSpanExporter - '(.+?)' : ([a-f0-9]+) ([a-f0-9]+) (\w+)"
)

services = ["carts", "orders", "shipping", "queue-master"]
containers = [f"docker-compose_{s}_1" for s in services]

procs = [
    subprocess.Popen(
        ["docker", "logs", "-f", "--tail=0", c],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    )
    for c in containers
]

print("Relaying OTel spans to LTTng... (Ctrl+C to stop)")
try:
    import select
    while True:
        readable, _, _ = select.select([p.stdout for p in procs], [], [], 1.0)
        for stream in readable:
            line = stream.readline().decode("utf-8", errors="replace").strip()
            m = pattern.search(line)
            if m:
                op, trace_id, span_id, kind = m.groups()
                msg = f"op={op} trace_id={trace_id} span_id={span_id} kind={kind}"
                logger.info(msg)
                print(f"[LTTng] {msg}")
except KeyboardInterrupt:
    print("Stopped.")
    for p in procs:
        p.terminate()
