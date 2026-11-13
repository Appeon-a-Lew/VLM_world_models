"""Local, dependency-free experiment visualization dashboard.

Serves episode data (RGB, segmentation, GT 3D boxes) together with VLM
predictions and evaluation metrics as a small single-page web app.  The HTTP
server uses only the Python standard library; the frontend is vanilla
HTML/CSS/JS with no external CDN or network fetches, so it works on offline
lab machines.

Usage
-----
::

    # Minimal: browse a data root (layout <root>/<split>/<episode_id>/meta.json)
    python3 scripts/dashboard.py --data /path/to/wm/data

    # Add one or more prediction jsonl files (name=path, repeatable)
    python3 scripts/dashboard.py --data /path/to/wm/data \
        --pred zeroshot=runs/zs/pred.jsonl \
        --pred sft=runs/sft/pred.jsonl

    # Also show evaluation results (results/table.md and/or results/metrics.csv)
    python3 scripts/dashboard.py --data /path/to/wm/data \
        --pred zeroshot=runs/zs/pred.jsonl --runs runs/exp001

    # Bind elsewhere / another port
    python3 scripts/dashboard.py --data ... --host 0.0.0.0 --port 8765

Open http://127.0.0.1:8765/ .  The hash route ``#<split>/<episode_id>`` is
shareable.  Prediction records are the jsonl rows written by
``physwm/vlm/infer.py`` (``episode_id``, ``task``, ``objects``, ``raw`` ...);
the ``objects`` values are camera-frame xyz centres.  Task A is compared at
t = 0 s and task B at t = 1 s, mirroring ``physwm.eval.metrics.TASK_TIME``.

Episode frame naming (both are discovered by scanning the episode directory,
so no fixed set is assumed):

- legacy: ``rgb_t<K>.png`` / ``seg_t<K>.png`` (single camera, ``meta.camera``)
- multi-view/frame: ``rgb_v<V>f<KK>.png`` / ``seg_v<V>f<KK>.png``
  (cameras in ``meta.views``, times in ``meta.frame_times``)

Endpoints
---------
``GET /api/splits``
``GET /api/episodes?split=S&limit=&offset=``
``GET /api/episode/<split>/<episode_id>``
``GET /api/preds``
``GET /api/pred/<name>/<split>/<episode_id>``
``GET /api/metrics``
``GET /file/<split>/<episode_id>/<name.png>``

Every JSON endpoint also answers with a trailing ``.json`` on the last path
segment (``/api/splits.json``, ``/api/episode/S/I.json``, ...).  That is the
canonical form used by ``dashboard_static/app.js``, which appends ``.json`` in
``fetchJSON``; the extensionless routes are kept for backwards compatibility.
The static bundle written by ``dashboard_export.py`` therefore mirrors the
server contract exactly (``api/splits.json``, ``api/pred/N/S/I.json``, ...).

Episode images are served under both ``/file/...`` and ``/files/...`` (same
handler and guards), so a static ``file_template`` of ``files/{split}/...``
works against the server too.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlsplit

STATIC_DIR = Path(__file__).resolve().parent / "dashboard_static"

_LEGACY_RGB = re.compile(r"^rgb_t(\d+)\.png$")
_MULTIFRAME_RGB = re.compile(r"^rgb_v(\d+)f(\d+)\.png$")

_STATIC_TYPES = {
    "index.html": "text/html; charset=utf-8",
    "app.js": "text/javascript; charset=utf-8",
    "style.css": "text/css; charset=utf-8",
}


class HttpError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status
        self.message = message


def _component(value: str, what: str) -> str:
    """Validate one path/URL component (no separators, no traversal)."""
    if not value or value in (".", "..") or "/" in value or "\\" in value or "\x00" in value:
        raise HttpError(HTTPStatus.BAD_REQUEST, f"invalid {what}")
    return value


class Dashboard:
    """Data access + in-memory caches behind the HTTP handler."""

    def __init__(self, root: Path, pred_files: dict[str, Path], runs_dir: Path | None):
        self.root = root.resolve()
        if not self.root.is_dir():
            raise SystemExit(f"--data is not a directory: {root}")
        self.runs = runs_dir.resolve() if runs_dir else None

        self._splits: list[dict] | None = None
        self._split_ids: dict[str, list[str]] = {}
        self._meta: dict[tuple[str, str], dict] = {}
        self._episodes: dict[tuple[str, str], dict] = {}
        self._summaries: dict[tuple[str, str], dict] = {}
        self._metrics: dict | None = None

        # Prediction records grouped by episode id.  A file may contain several
        # tasks / repeats per episode; all rows are kept for the raw viewer.
        self.pred_names = list(pred_files)
        self.preds: dict[str, dict[str, list[dict]]] = {}
        for name, path in pred_files.items():
            self.preds[name] = self._load_predictions(name, path)

    # -- predictions -----------------------------------------------------
    @staticmethod
    def _load_predictions(name: str, path: Path) -> dict[str, list[dict]]:
        records: dict[str, list[dict]] = {}
        try:
            with open(path, encoding="utf-8") as f:
                for lineno, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise SystemExit(f"bad json in {name}={path} line {lineno}: {exc}") from exc
                    eid = rec.get("episode_id")
                    if eid is not None:
                        records.setdefault(str(eid), []).append(rec)
        except FileNotFoundError:
            raise SystemExit(f"prediction file not found: {path}") from None
        return records

    # -- splits / episodes ----------------------------------------------
    def splits(self) -> list[dict]:
        if self._splits is None:
            out = []
            for entry in sorted(self.root.iterdir()):
                if not entry.is_dir() or entry.name.startswith("."):
                    continue
                ids = self._episode_ids(entry.name)
                if ids:
                    out.append({"split": entry.name, "n_episodes": len(ids)})
            self._splits = out
        return self._splits

    def _episode_ids(self, split: str) -> list[str]:
        """Episode ids of a split, honouring index.jsonl when it exists."""
        if split in self._split_ids:
            return self._split_ids[split]
        split_dir = self.root / split
        if not split_dir.is_dir():
            self._split_ids[split] = []
            return []
        index = split_dir / "index.jsonl"
        if index.is_file():
            ids = []
            for line in index.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    ids.append(str(json.loads(line)["episode_id"]))
                except (json.JSONDecodeError, KeyError, TypeError):
                    continue
            ids = sorted({i for i in ids if (split_dir / i / "meta.json").is_file()})
        else:
            ids = sorted(p.parent.name for p in split_dir.glob("*/meta.json") if p.is_file())
        self._split_ids[split] = ids
        return ids

    def _meta_for(self, split: str, episode_id: str) -> dict:
        key = (split, episode_id)
        if key not in self._meta:
            _component(split, "split")
            _component(episode_id, "episode id")
            path = self.root / split / episode_id / "meta.json"
            if not path.is_file():
                raise HttpError(HTTPStatus.NOT_FOUND, f"no such episode: {split}/{episode_id}")
            self._meta[key] = json.loads(path.read_text(encoding="utf-8"))
        return self._meta[key]

    def _scan_frames(self, split: str, episode_id: str, meta: dict) -> list[dict]:
        """Discover rgb/seg files and group them into [(t, view)] frames."""
        ep_dir = self.root / split / episode_id
        by_index: dict[int, dict[int, dict]] = {}
        for path in ep_dir.iterdir():
            if not path.is_file():
                continue
            m = _LEGACY_RGB.match(path.name)
            view = 0
            if m:
                k = int(m.group(1))
            else:
                m = _MULTIFRAME_RGB.match(path.name)
                if not m:
                    continue
                view, k = int(m.group(1)), int(m.group(2))
            seg_name = "seg" + path.name[3:]  # rgb_... -> seg_...
            seg = seg_name if (ep_dir / seg_name).is_file() else None
            by_index.setdefault(k, {})[view] = {"view": view, "rgb": path.name, "seg": seg}

        frame_times = meta.get("frame_times")
        times = meta.get("times")
        frames = []
        for k in sorted(by_index):
            if isinstance(frame_times, list) and k < len(frame_times):
                t = float(frame_times[k])
            elif isinstance(times, list) and k < len(times):
                t = float(times[k])
            else:
                t = float(k)
            views = [by_index[k][v] for v in sorted(by_index[k])]
            frames.append({"k": k, "t": t, "views": views})
        return frames

    def _camera_views(self, meta: dict) -> list[dict]:
        views = meta.get("views")
        if isinstance(views, list) and views:
            return [v for v in views if isinstance(v, dict)]
        camera = meta.get("camera")
        return [camera] if isinstance(camera, dict) else []

    def episode_payload(self, split: str, episode_id: str) -> dict:
        key = (split, episode_id)
        if key in self._episodes:
            return self._episodes[key]
        meta = self._meta_for(split, episode_id)
        frames = self._scan_frames(split, episode_id, meta)
        payload = dict(meta)
        payload["views"] = self._camera_views(meta)
        payload["frames"] = frames
        payload["n_frames"] = len(frames)
        payload["preds"] = list(self.pred_names)
        payload["available_preds"] = list(self.pred_names)
        payload["thumb"] = frames[0]["views"][0]["rgb"] if frames and frames[0]["views"] else None
        self._episodes[key] = payload
        return payload

    def episode_summary(self, split: str, episode_id: str) -> dict:
        key = (split, episode_id)
        if key in self._summaries:
            return self._summaries[key]
        meta = self._meta_for(split, episode_id)
        frames = self._scan_frames(split, episode_id, meta)
        cameras = self._camera_views(meta)
        camera = cameras[0] if cameras else {}
        action = meta.get("action") or {}
        summary = {
            "episode_id": str(meta.get("episode_id", episode_id)),
            "n_objects": len(meta.get("objects") or []),
            "action_text": str(action.get("text", "")),
            "elevation_deg": camera.get("elevation_deg"),
            "n_frames": len(frames),
            "n_views": len(cameras),
            "thumb": frames[0]["views"][0]["rgb"] if frames and frames[0]["views"] else None,
        }
        self._summaries[key] = summary
        return summary

    def metrics(self) -> dict:
        if self._metrics is not None:
            return self._metrics
        table_md = None
        metrics_csv = None
        if self.runs:
            for candidate in (self.runs / "results" / "table.md", self.runs / "table.md"):
                if candidate.is_file():
                    table_md = candidate.read_text(encoding="utf-8", errors="replace")
                    break
            for candidate in (self.runs / "results" / "metrics.csv", self.runs / "metrics.csv"):
                if candidate.is_file():
                    with open(candidate, newline="", encoding="utf-8", errors="replace") as f:
                        metrics_csv = list(csv.reader(f))
                    break
        self._metrics = {"table_md": table_md, "metrics_csv": metrics_csv}
        return self._metrics


class Handler(BaseHTTPRequestHandler):
    server_version = "WMDashboard/1.0"

    @property
    def app(self) -> Dashboard:
        return self.server.app  # type: ignore[attr-defined]

    def do_GET(self) -> None:  # noqa: N802 (http.server API)
        self._dispatch(head=False)

    def do_HEAD(self) -> None:  # noqa: N802
        self._dispatch(head=True)

    def _dispatch(self, head: bool) -> None:
        parsed = urlsplit(self.path)
        parts = [unquote(p) for p in parsed.path.split("/") if p]
        try:
            if not parts:
                self._static("index.html", head)
            elif parts[0] in ("file", "files"):
                self._episode_file(parts, head)
            elif parts[0] == "api":
                self._api(parts, parse_qs(parsed.query), head)
            elif len(parts) == 1 and parts[0] in _STATIC_TYPES:
                self._static(parts[0], head)
            else:
                raise HttpError(HTTPStatus.NOT_FOUND, "not found")
        except HttpError as exc:
            self._send_json({"error": exc.message}, status=exc.status, head=head)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as exc:  # surface server bugs as JSON, not a dead socket
            self.log_error("unhandled error: %r", exc)
            self._send_json({"error": f"internal error: {exc}"}, status=500, head=head)

    # -- routing ---------------------------------------------------------
    def _api(self, parts: list[str], query: dict, head: bool) -> None:
        # The frontend always requests ``<endpoint>.json``; the extension is
        # optional so both /api/splits and /api/splits.json work.
        if len(parts) > 1 and parts[-1].endswith(".json"):
            parts = parts[:-1] + [parts[-1][: -len(".json")]]
        if parts == ["api", "splits"]:
            self._send_json(self.app.splits(), head=head)
        elif parts == ["api", "episodes"]:
            split = (query.get("split") or [""])[0]
            split = _component(split, "split")
            ids = self.app._episode_ids(split)
            if not ids:
                raise HttpError(HTTPStatus.NOT_FOUND, f"unknown or empty split: {split}")
            offset = self._int_query(query, "offset", 0)
            limit = self._int_query(query, "limit", None)
            selected = ids[offset:] if limit is None else ids[offset:offset + limit]
            self._send_json([self.app.episode_summary(split, eid) for eid in selected], head=head)
        elif len(parts) == 4 and parts[:2] == ["api", "episode"]:
            self._send_json(self.app.episode_payload(parts[2], parts[3]), head=head)
        elif parts == ["api", "preds"]:
            self._send_json(self.app.pred_names, head=head)
        elif len(parts) == 5 and parts[:2] == ["api", "pred"]:
            name, split, episode_id = parts[2], parts[3], parts[4]
            if name not in self.app.preds:
                raise HttpError(HTTPStatus.NOT_FOUND, f"unknown prediction source: {name}")
            _component(split, "split")
            _component(episode_id, "episode id")
            records = self.app.preds[name].get(episode_id, [])
            self._send_json(records, head=head)
        elif parts == ["api", "metrics"]:
            self._send_json(self.app.metrics(), head=head)
        else:
            raise HttpError(HTTPStatus.NOT_FOUND, "unknown API endpoint")

    def _episode_file(self, parts: list[str], head: bool) -> None:
        if len(parts) != 4:
            raise HttpError(HTTPStatus.BAD_REQUEST, "expected /file|/files/<split>/<id>/<name>")
        split = _component(parts[1], "split")
        episode_id = _component(parts[2], "episode id")
        name = _component(parts[3], "file name")
        if not name.lower().endswith(".png") or name.startswith("."):
            raise HttpError(HTTPStatus.BAD_REQUEST, "only .png files are served")
        ep_dir = (self.app.root / split / episode_id).resolve()
        path = (ep_dir / name).resolve()
        if path.parent != ep_dir or not path.is_file():
            raise HttpError(HTTPStatus.NOT_FOUND, f"no such file: {split}/{episode_id}/{name}")
        data = path.read_bytes()
        self._send_bytes(data, "image/png", head, cache=True)

    # -- responses -------------------------------------------------------
    def _static(self, name: str, head: bool) -> None:
        path = STATIC_DIR / name
        if not path.is_file():
            raise HttpError(HTTPStatus.NOT_FOUND, f"missing static file: {name}")
        self._send_bytes(path.read_bytes(), _STATIC_TYPES[name], head)

    def _send_json(self, obj, status: int = 200, head: bool = False) -> None:
        body = json.dumps(obj).encode("utf-8")
        self._send_bytes(body, "application/json; charset=utf-8", head, status=status)

    def _send_bytes(self, body: bytes, content_type: str, head: bool,
                    status: int = 200, cache: bool = False) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "public, max-age=3600" if cache else "no-store")
        self.end_headers()
        if not head:
            self.wfile.write(body)

    @staticmethod
    def _int_query(query: dict, name: str, default: int | None) -> int | None:
        raw = (query.get(name) or [None])[0]
        if raw is None or raw == "":
            return default
        try:
            value = int(raw)
        except ValueError:
            raise HttpError(HTTPStatus.BAD_REQUEST, f"{name} must be an integer")
        if value < 0:
            raise HttpError(HTTPStatus.BAD_REQUEST, f"{name} must be >= 0")
        return value

    def log_message(self, fmt: str, *args) -> None:
        sys.stderr.write("[dashboard] %s\n" % (fmt % args))


def _parse_pred(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("expected NAME=PATH")
    name, path = value.split("=", 1)
    if not name or "/" in name or "\\" in name or name in (".", ".."):
        raise argparse.ArgumentTypeError(f"invalid prediction name: {name!r}")
    if not path:
        raise argparse.ArgumentTypeError("empty prediction path")
    return name, Path(path)


def main() -> None:
    ap = argparse.ArgumentParser(description="Local experiment visualization dashboard.")
    ap.add_argument("--data", required=True, help="data root containing <split>/<episode_id>/meta.json")
    ap.add_argument("--pred", action="append", default=[], type=_parse_pred, metavar="NAME=PATH.jsonl",
                    help="prediction jsonl file, repeatable")
    ap.add_argument("--runs", default=None, help="run dir with results/table.md and/or metrics.csv")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()

    pred_files: dict[str, Path] = {}
    for name, path in args.pred:
        if name in pred_files:
            ap.error(f"duplicate prediction name: {name}")
        pred_files[name] = path

    app = Dashboard(Path(args.data), pred_files, Path(args.runs) if args.runs else None)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.daemon_threads = True
    server.app = app  # type: ignore[attr-defined]
    url = f"http://{args.host}:{args.port}/"
    print(f"dashboard serving {app.root} at {url}")
    print(f"  splits: {', '.join(s['split'] for s in app.splits()) or '(none)'}")
    if pred_files:
        print(f"  predictions: {', '.join(pred_files)}")
    else:
        print("  predictions: (none; pass --pred NAME=PATH.jsonl)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()