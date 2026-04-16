import json
import re
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn
from typing import Any, Dict, Optional, Tuple
from urllib.parse import parse_qs, urlparse

from .errors import SandboxError
from .manager import SandboxManager

try:
    from http.server import ThreadingHTTPServer
except ImportError:
    class ThreadingHTTPServer(ThreadingMixIn, HTTPServer):  # type: ignore[misc]
        daemon_threads = True


class SandboxHTTPServer(ThreadingHTTPServer):
    def __init__(self, server_address: Tuple[str, int], manager: SandboxManager):
        super().__init__(server_address, SandboxRequestHandler)
        self.manager = manager


class SandboxRequestHandler(BaseHTTPRequestHandler):
    server: SandboxHTTPServer

    def _send_json(self, status: int, payload: Dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=True).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _parse_json_body(self) -> Dict[str, Any]:
        length_header = self.headers.get("Content-Length")
        if not length_header:
            return {}
        try:
            length = int(length_header)
        except ValueError as exc:
            raise SandboxError("Invalid Content-Length") from exc
        if length <= 0:
            return {}
        raw = self.rfile.read(length)
        if not raw:
            return {}
        try:
            payload = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise SandboxError(f"Invalid JSON body: {exc.msg}") from exc
        if not isinstance(payload, dict):
            raise SandboxError("JSON body must be an object")
        return payload

    def _extract_env_route(self, path: str) -> Optional[Tuple[str, str]]:
        match = re.fullmatch(r"/envs/([a-zA-Z0-9_-]+)(?:/([a-zA-Z0-9_-]+))?", path)
        if not match:
            return None
        env_id = match.group(1)
        action = match.group(2) or ""
        return env_id, action

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path
        try:
            if path == "/health":
                self._send_json(200, {"status": "ok", "service": "skillsbench-sandbox"})
                return
            if path == "/envs":
                self._send_json(200, {"envs": self.server.manager.list_envs()})
                return

            route = self._extract_env_route(path)
            if route:
                env_id, action = route
                if action == "":
                    self._send_json(200, self.server.manager.get_env(env_id))
                    return
                if action == "instruction":
                    self._send_json(200, self.server.manager.get_instruction(env_id))
                    return
            raise SandboxError(f"Unknown endpoint: {path}", 404)
        except SandboxError as err:
            self._send_json(err.status_code, {"error": err.message})
        except Exception as exc:
            self._send_json(500, {"error": f"Internal server error: {exc}"})

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path
        try:
            payload = self._parse_json_body()
            if path == "/envs":
                data = self.server.manager.create_env(payload)
                self._send_json(201, data)
                return

            route = self._extract_env_route(path)
            if route:
                env_id, action = route
                if action == "step":
                    self._send_json(200, self.server.manager.step(env_id, payload))
                    return
                if action == "evaluate":
                    self._send_json(200, self.server.manager.evaluate(env_id, payload))
                    return
                if action == "reset":
                    self._send_json(200, self.server.manager.reset(env_id))
                    return
            raise SandboxError(f"Unknown endpoint: {path}", 404)
        except SandboxError as err:
            self._send_json(err.status_code, {"error": err.message})
        except Exception as exc:
            self._send_json(500, {"error": f"Internal server error: {exc}"})

    @staticmethod
    def _extract_image_route(path: str) -> Optional[str]:
        """Match /images/<image_tag> where tag may contain colons, dots, dashes."""
        match = re.fullmatch(r"/images/([a-zA-Z0-9_.:-]+)", path)
        return match.group(1) if match else None

    def do_DELETE(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path
        try:
            # DELETE /images/<image_tag>
            image_tag = self._extract_image_route(path)
            if image_tag:
                self._send_json(200, self.server.manager.remove_image(image_tag))
                return

            # DELETE /envs/<env_id>
            route = self._extract_env_route(path)
            if not route:
                raise SandboxError(f"Unknown endpoint: {path}", 404)
            env_id, action = route
            if action not in ("", "delete"):
                raise SandboxError(f"Unknown endpoint: {path}", 404)
            query = parse_qs(parsed.query)
            remove_image = query.get("remove_image", ["false"])[0].lower() in {"1", "true", "yes"}
            self._send_json(200, self.server.manager.delete_env(env_id, remove_image=remove_image))
        except SandboxError as err:
            self._send_json(err.status_code, {"error": err.message})
        except Exception as exc:
            self._send_json(500, {"error": f"Internal server error: {exc}"})

    def log_message(self, fmt: str, *args: Any) -> None:
        message = fmt % args
        print(f"[sandbox] {self.client_address[0]} {self.command} {self.path} -> {message}")
