import hashlib
import secrets
import threading
import time
from collections import deque
from urllib.parse import urlsplit


LOCAL_FRONTEND_ORIGINS = (
    "http://localhost:5173",
    "http://127.0.0.1:5173",
)


def normalize_origin(origin: str) -> str:
    value = origin.strip()
    parsed = urlsplit(value)
    if (
        parsed.scheme.lower() not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path not in {"", "/"}
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError(f"Invalid frontend origin: {origin!r}")

    scheme = parsed.scheme.lower()
    hostname = parsed.hostname.lower()
    is_loopback = hostname in {"localhost", "127.0.0.1", "::1"}
    if scheme != "https" and not is_loopback:
        raise ValueError("Non-local frontend origins must use HTTPS.")
    if parsed.port is not None and not 1 <= parsed.port <= 65535:
        raise ValueError(f"Invalid frontend origin port: {origin!r}")

    default_port = (scheme == "https" and parsed.port == 443) or (
        scheme == "http" and parsed.port == 80
    )
    host = f"[{hostname}]" if ":" in hostname else hostname
    port = f":{parsed.port}" if parsed.port and not default_port else ""
    return f"{scheme}://{host}{port}"


def configured_frontend_origins(value: str | None) -> list[str]:
    configured = [
        normalize_origin(item)
        for item in (value or "").split(",")
        if item.strip()
    ]
    return list(dict.fromkeys((*LOCAL_FRONTEND_ORIGINS, *configured)))


def validate_access_password(password: str) -> None:
    if password and not password.strip():
        raise ValueError("RAG_ACCESS_PASSWORD must not be blank.")
    if password and len(password) < 16:
        raise ValueError("RAG_ACCESS_PASSWORD must contain at least 16 characters.")


class AccessTokenManager:
    def __init__(self, ttl_seconds: int) -> None:
        self.ttl_seconds = ttl_seconds
        self._sessions: dict[str, float] = {}
        self._lock = threading.Lock()

    def issue(self, supplied_password: str, configured_password: str) -> tuple[str, int]:
        if not secrets.compare_digest(
            supplied_password.encode("utf-8"),
            configured_password.encode("utf-8"),
        ):
            raise ValueError("Invalid access password.")

        token = secrets.token_urlsafe(32)
        expires_at = int(time.time()) + self.ttl_seconds
        token_hash = hashlib.sha256(token.encode("utf-8")).hexdigest()
        with self._lock:
            self._remove_expired()
            self._sessions[token_hash] = float(expires_at)
        return token, expires_at

    def validate(self, token: str) -> bool:
        token_hash = hashlib.sha256(token.encode("utf-8")).hexdigest()
        now = time.time()
        with self._lock:
            expires_at = self._sessions.get(token_hash)
            if expires_at is None:
                self._remove_expired(now)
                return False
            if expires_at <= now:
                del self._sessions[token_hash]
                return False
            return True

    def revoke(self, token: str) -> bool:
        token_hash = hashlib.sha256(token.encode("utf-8")).hexdigest()
        with self._lock:
            return self._sessions.pop(token_hash, None) is not None

    def _remove_expired(self, now: float | None = None) -> None:
        current_time = now if now is not None else time.time()
        expired = [
            token_hash
            for token_hash, expires_at in self._sessions.items()
            if expires_at <= current_time
        ]
        for token_hash in expired:
            del self._sessions[token_hash]


class InMemoryRateLimiter:
    def __init__(self) -> None:
        self._requests: dict[tuple[str, str], deque[float]] = {}
        self._lock = threading.Lock()

    def allow(
        self,
        key: str,
        bucket: str,
        *,
        limit: int,
        window_seconds: int,
    ) -> tuple[bool, int]:
        now = time.monotonic()
        cutoff = now - window_seconds
        request_key = (key, bucket)
        with self._lock:
            requests = self._requests.setdefault(request_key, deque())
            while requests and requests[0] <= cutoff:
                requests.popleft()
            if len(requests) >= limit:
                return False, max(1, int(window_seconds - (now - requests[0])))
            requests.append(now)
            return True, 0
