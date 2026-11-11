"""Thin OpenAI-compatible proxy in front of the shared vLLM server, used by the OFFICIAL baseline code.
 - the baselines call the OpenAI API without any chat-template option, so the proxy switches Qwen3's thinking off
   (the same setting every other method of this project uses); nothing else in the request is touched;
 - it counts calls and tokens per tag.  A tag is the path prefix of the base URL:  http://host:PORT/<tag>/v1 ,
   so a worker obtains the cost of one phase (build / correction S1 / ...) of one memory by reading /_usage/<tag>.
Usage: python llm_proxy.py PORT UPSTREAM_PORT"""
import http.client
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PORT, UP = int(sys.argv[1]), int(sys.argv[2])
STRIP = "--strip-schema" in sys.argv     # drop json_schema response formats: vLLM (V0) serves them with the outlines backend, which is
                                         # single-threaded CPU decoding and stalls the whole engine; the prompts still ask for JSON
USAGE, LOCK = {}, threading.Lock()


class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype="application/json"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.startswith("/_usage/"):
            tag = self.path[len("/_usage/"):]
            with LOCK:
                u = dict(USAGE.get(tag, {"calls": 0, "in": 0, "out": 0}))
            return self._send(200, json.dumps(u).encode())
        c = http.client.HTTPConnection("localhost", UP, timeout=600)
        c.request("GET", "/v1" + self.path.partition("/v1")[2])
        r = c.getresponse()
        self._send(r.status, r.read())

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(n)
        tag, _, rest = self.path[1:].partition("/v1")
        try:
            js = json.loads(raw)
            js.setdefault("chat_template_kwargs", {})["enable_thinking"] = False
            js.pop("stream", None)
            if STRIP and (js.get("response_format") or {}).get("type") == "json_schema":
                js.pop("response_format")
            body = json.dumps(js).encode()
        except Exception:
            body = raw
        for attempt in range(3):
            try:
                c = http.client.HTTPConnection("localhost", UP, timeout=3600)
                c.request("POST", "/v1" + rest, body, {"Content-Type": "application/json"})
                r = c.getresponse()
                data = r.read()
                break
            except Exception:
                if attempt == 2:
                    return self._send(502, b'{"error": "upstream"}')
        try:
            u = json.loads(data).get("usage") or {}
            with LOCK:
                t = USAGE.setdefault(tag, {"calls": 0, "in": 0, "out": 0})
                t["calls"] += 1
                t["in"] += u.get("prompt_tokens", 0)
                t["out"] += u.get("completion_tokens", 0)
        except Exception:
            pass
        self._send(r.status, data)


class Server(ThreadingHTTPServer):
    request_queue_size = 1024
    daemon_threads = True


if __name__ == "__main__":
    Server(("0.0.0.0", PORT), H).serve_forever()
