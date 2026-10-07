"""Fixed diagnostic providers. Offline outputs are heuristics, never Jev results."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
import threading
import time
from dataclasses import asdict, dataclass
from contextlib import contextmanager
from urllib import error, request

CAUSES = ("credentials", "connections", "dependency")
DEFAULT_MODEL = "jev-1.13.0"
API_URL = "https://api.typesafe.ai/v1/systemone"
RUBRIC_VERSION = "incident-causes-v1"


def redact_logs(logs: str) -> str:
    """Remove common credential forms before either persistence or transport.

    This is a bounded best-effort filter, not a guarantee that arbitrary logs are
    safe to send. Callers should collect a deliberately restricted excerpt.
    """
    logs = re.sub(r"(?i)(authorization[\"']?\s*:\s*[\"']?bearer\s+)[^\s\"']+", r"\1[REDACTED]", logs)
    logs = re.sub(r"(?i)([\"']?\b(?:password|passwd|pwd|token|api[_-]?key|secret)\b[\"']?\s*[=:]\s*)(?:\"[^\"]*\"|'[^']*'|[^\s,;}]+)",
                  r'\1"[REDACTED]"', logs)
    return re.sub(r"(\b[a-z][a-z0-9+.-]*://)[^\s/@]+:[^\s/@]+@", r"\1[REDACTED]@", logs,
                  flags=re.I)


def _checksum(record: dict) -> str:
    clean = {key: value for key, value in record.items() if key != "checksum"}
    return hashlib.sha256(json.dumps(clean, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class Diagnosis:
    probabilities: dict[str, float]
    source: str
    model: str
    input_tokens: int
    estimated_cost_usd: float
    cache_hit: bool
    latency_ms: float = 0.0

    @property
    def ranked_causes(self) -> list[str]:
        """Return predefined causes by descending likelihood; ties are stable."""
        return sorted(CAUSES, key=lambda cause: -self.probabilities[cause])


class DiagnosticError(RuntimeError):
    """A diagnostic could not be obtained without changing provider semantics."""


class _NoRedirect(request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise DiagnosticError("Jev endpoint redirects are refused to protect credentials")


def _safe_urlopen(call, timeout):
    return request.build_opener(_NoRedirect()).open(call, timeout=timeout)


_PATH_LOCKS: dict[str, threading.RLock] = {}
_PATH_LOCKS_GUARD = threading.Lock()


@contextmanager
def _cache_guard(path: Path | None, readonly: bool):
    if path is None:
        yield
        return
    with _PATH_LOCKS_GUARD:
        lock = _PATH_LOCKS.setdefault(str(path.resolve()), threading.RLock())
    with lock:
        if readonly:
            yield
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.with_suffix(path.suffix + '.lock').open('a') as lockfile:
            try:
                import fcntl
            except ImportError:  # Windows still has the shared in-process lock.
                fcntl = None
            if fcntl is not None:
                fcntl.flock(lockfile.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                if fcntl is not None:
                    fcntl.flock(lockfile.fileno(), fcntl.LOCK_UN)


def _probabilities(values: object) -> dict[str, float]:
    if not isinstance(values, dict):
        raise DiagnosticError("Diagnosis must contain named cause probabilities")
    result = {}
    for cause in CAUSES:
        value = values.get(cause)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise DiagnosticError(f"Missing or nonnumeric probability: {cause}")
        value = float(value)
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise DiagnosticError(f"Probability outside [0, 1]: {cause}")
        result[cause] = value
    return result


class DiagnosticProvider:
    """Use ``live`` explicitly to call hosted Jev; ``cached`` never uses network.

    Costs are fresh-call estimates for input tokens only. Cache hits cost zero;
    the original usage remains attached for provenance. Cause scores are
    independent because several injected faults may coexist.
    """

    def __init__(self, mode: str = "offline", cache_path: str | Path | None = None,
                 model: str = DEFAULT_MODEL, input_cost_per_million: float = 0.042,
                 max_log_chars: int = 4096, timeout_seconds: float = 10.0,
                 max_attempts: int = 2, max_live_calls: int = 1000,
                 max_api_cost_usd: float = 1.0):
        if mode not in {"offline", "live", "cached"}:
            raise ValueError("mode must be offline, live, or cached")
        if not isinstance(model, str) or not re.fullmatch(r"jev-\d+\.\d+\.\d+", model):
            raise ValueError("A pinned model version such as jev-1.13.0 is required")
        if not math.isfinite(input_cost_per_million) or input_cost_per_million < 0:
            raise ValueError("Input cost must be finite and nonnegative")
        for name, value in {"max_log_chars": max_log_chars, "max_attempts": max_attempts,
                            "max_live_calls": max_live_calls}.items():
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be finite and positive")
        if not math.isfinite(max_api_cost_usd) or max_api_cost_usd <= 0:
            raise ValueError("max_api_cost_usd must be finite and positive")
        self.statistics = {'hosted_calls': 0, 'cache_hits': 0, 'input_tokens': 0,
                           'estimated_api_cost_usd': 0.0, 'offline_computations': 0,
                           'network_attempts': 0, 'retries': 0, 'failures': 0,
                           'redacted_requests': 0, 'latency_ms': 0.0}
        self.mode = mode
        self.model = model
        self.input_cost_per_million = input_cost_per_million
        self.max_log_chars, self.timeout_seconds = max_log_chars, timeout_seconds
        self.max_attempts, self.max_live_calls = max_attempts, max_live_calls
        self.max_api_cost_usd = max_api_cost_usd
        self._lock = threading.RLock()
        self._cache = {}
        self._cache_stamp = None
        self.cache_path = Path(cache_path) if cache_path is not None else None
        if self.cache_path is not None and not self.cache_path.is_absolute():
            raise ValueError("cache_path must be absolute")

    def _payload(self, logs: str) -> dict:
        instructions = {
            "credentials": "Estimate likelihood that logs indicate invalid database credentials or authentication configuration.",
            "connections": "Estimate likelihood that logs indicate exhausted database connections or a saturated connection pool.",
            "dependency": "Estimate likelihood that logs indicate a failed or unreachable upstream dependency.",
        }
        return {"model": self.model, "state": logs, "questions": {
            cause: {"type": "noul", "instructions": instructions[cause]}
            for cause in CAUSES
        }}

    def _key(self, payload: dict, namespace: str) -> str:
        identity = {"namespace": namespace, "rubric": RUBRIC_VERSION, "request": payload}
        return hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    def _read_cache(self) -> dict:
        if self.cache_path is None or not self.cache_path.exists():
            return {}
        try:
            stat = self.cache_path.stat()
            stamp = (stat.st_mtime_ns, stat.st_size)
            if stamp == self._cache_stamp:
                return self._cache
            contents = json.loads(self.cache_path.read_text())
            if not isinstance(contents, dict):
                raise ValueError("cache root must be an object")
            self._cache, self._cache_stamp = contents, stamp
            return self._cache
        except (OSError, ValueError) as exc:
            raise DiagnosticError("Unable to read diagnostic cache") from exc

    def diagnose(self, logs: str) -> Diagnosis:
        with self._lock, _cache_guard(self.cache_path, self.mode == 'cached'):
            started = time.monotonic()
            try:
                return self._diagnose(logs)
            except DiagnosticError:
                self.statistics['failures'] += 1
                raise
            except OSError as exc:
                self.statistics['failures'] += 1
                raise DiagnosticError("Unable to persist diagnostic cache") from exc
            finally:
                self.statistics['latency_ms'] += (time.monotonic() - started) * 1000

    def _diagnose(self, logs: str) -> Diagnosis:
        if not isinstance(logs, str):
            raise TypeError("logs must be a compact string, without simulator truth")
        if len(logs) > self.max_log_chars:
            raise DiagnosticError("Log excerpt exceeds the diagnostic size budget")
        clean_logs = redact_logs(logs)
        self.statistics['redacted_requests'] += int(clean_logs != logs)
        logs = clean_logs
        payload = self._payload(logs)
        namespace = "offline" if self.mode == "offline" else "live"
        key = self._key(payload, namespace)
        cache = self._read_cache()
        if key in cache:
            record = cache[key]
            # Old offline entries can be recomputed without trusting their values.
            if self.mode == 'offline' and isinstance(record, dict) and 'checksum' not in record:
                del cache[key]
                return self._diagnose(logs)
            try:
                if not isinstance(record, dict):
                    raise ValueError("cache entry must be an object")
                if record.get('checksum') != _checksum(record):
                    raise ValueError("cache integrity mismatch")
                if record['request'] != payload or record['rubric'] != RUBRIC_VERSION:
                    raise ValueError("cached request/rubric mismatch")
                data = record["diagnosis"]
                if record["requested_model"] != self.model:
                    raise ValueError("model mismatch")
                expected_source = "offline-heuristic" if namespace == "offline" else "hosted-jev"
                if data["source"] != expected_source:
                    raise ValueError("provider provenance mismatch")
                if not isinstance(record["resolved_model"], str) or not record["resolved_model"]:
                    raise ValueError("invalid resolved model")
                if namespace == "live" and record["resolved_model"] != self.model:
                    raise ValueError("resolved model differs from requested pin")
                tokens = self._tokens(data["input_tokens"])
                probabilities = _probabilities(data["probabilities"])
                self.statistics['cache_hits'] += 1
                return Diagnosis(probabilities, data["source"],
                                 record["resolved_model"], tokens, 0.0, True)
            except (KeyError, TypeError, ValueError) as exc:
                raise DiagnosticError("Invalid diagnostic cache record") from exc
        if self.mode == "cached":
            raise DiagnosticError("Hosted Jev cache miss; cached mode never calls the API")
        if self.mode == 'live' and (self.statistics['hosted_calls'] >= self.max_live_calls
                                   or self.statistics['estimated_api_cost_usd'] >= self.max_api_cost_usd
                                   or self.statistics['network_attempts'] >= self.max_live_calls * self.max_attempts):
            raise DiagnosticError("Hosted diagnostic call or dollar budget exhausted")
        # A conservative admission reservation (one token per UTF-8 payload byte
        # plus 1024 framing tokens), not an invoice guarantee. Exact returned
        # usage is counted after completion; retries may incur unreported billing.
        reserved_usd = (len(json.dumps(payload).encode()) + 1024) * self.input_cost_per_million / 1_000_000
        if self.mode == 'live' and self.statistics['estimated_api_cost_usd'] + reserved_usd > self.max_api_cost_usd:
            raise DiagnosticError("Hosted diagnostic request exceeds reserved dollar budget")
        diagnosis = self._offline(logs) if self.mode == "offline" else self._live(payload)
        if self.mode == 'live':
            self.statistics['hosted_calls'] += 1
            self.statistics['input_tokens'] += diagnosis.input_tokens
            self.statistics['estimated_api_cost_usd'] += diagnosis.estimated_cost_usd
        else:
            self.statistics['offline_computations'] += 1
        if self.cache_path is not None:
            cache[key] = {"requested_model": self.model, "resolved_model": diagnosis.model,
                          "rubric": RUBRIC_VERSION, "request": payload,
                          "diagnosis": asdict(diagnosis)}
            cache[key]['checksum'] = _checksum(cache[key])
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(mode='w', dir=self.cache_path.parent,
                                                 prefix='.diagnostics-', delete=False) as file:
                    temporary = Path(file.name)
                    json.dump(cache, file, sort_keys=True, allow_nan=False)
                    file.flush()
                    os.fsync(file.fileno())
                temporary.replace(self.cache_path)
                stat = self.cache_path.stat()
                self._cache, self._cache_stamp = cache, (stat.st_mtime_ns, stat.st_size)
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
        return diagnosis

    @staticmethod
    def _tokens(value: object) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise DiagnosticError("Invalid input token usage")
        return value

    def _offline(self, logs: str) -> Diagnosis:
        patterns = {
            "credentials": r"authenticat|credential|password|login|access denied|permission denied|secret|unauthori[sz]ed|identity rejected",
            "connections": r"too many (?:clients|connections)|pool|connection limit|connection slots|exhaust|saturat|capacity|no available connections|remaining connection|checkout.*timed out",
            "dependency": r"upstream|dependency|connection refused|unreachable|dns|name resolution|host.*down|service unavailable|gateway|endpoint|backend unavailable",
        }
        scores = {cause: 0.85 if re.search(pattern, logs, re.I) else 0.08
                  for cause, pattern in patterns.items()}
        return Diagnosis(scores, "offline-heuristic", "offline-heuristic-v1", 0, 0.0, False)

    def _live(self, payload: dict) -> Diagnosis:
        api_key = os.environ.get("TYPESAFE_API_KEY")
        if not api_key:
            raise DiagnosticError("Live Jev requires TYPESAFE_API_KEY")
        call = request.Request(API_URL, data=json.dumps(payload).encode(), method="POST",
                               headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"})
        started = time.monotonic()
        for attempt in range(self.max_attempts):
            self.statistics['network_attempts'] += 1
            try:
                with _safe_urlopen(call, timeout=self.timeout_seconds) as response:
                    body = response.read(1_048_577)
                    if len(body) > 1_048_576:
                        raise DiagnosticError("Jev response exceeds maximum size")
                    data = json.loads(body)
                break
            except error.HTTPError as exc:
                exc.close()
                if exc.code not in (429, 502, 503, 504) or attempt + 1 == self.max_attempts:
                    raise DiagnosticError(f"Jev API HTTP {exc.code}") from exc
            except (error.URLError, TimeoutError) as exc:
                if attempt + 1 == self.max_attempts:
                    raise DiagnosticError("Jev API request failed") from exc
            except (ValueError, UnicodeError) as exc:
                raise DiagnosticError("Malformed Jev API response") from exc
            self.statistics['retries'] += 1
            time.sleep(min(.2 * 2 ** attempt, 1.0))
        try:
            probabilities = _probabilities({cause: data["answers"][cause]["noul"] for cause in CAUSES})
            usage = data["usage"]
            tokens = self._tokens(usage.get("input_tokens", usage.get("prompt_tokens")))
            if "output_tokens" in usage:
                self._tokens(usage["output_tokens"])
            resolved_model = data["model"]
            if resolved_model != self.model:
                raise DiagnosticError("Hosted resolved model differs from requested pin")
        except (KeyError, TypeError, AttributeError) as exc:
            raise DiagnosticError("Malformed Jev API response") from exc
        return Diagnosis(probabilities, "hosted-jev", resolved_model, tokens,
                         tokens * self.input_cost_per_million / 1_000_000, False,
                         (time.monotonic() - started) * 1000)
