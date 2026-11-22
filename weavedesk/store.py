import contextlib
import hashlib
import json
import os
import sqlite3
from pathlib import Path
from uuid import uuid4

from .model import ConflictError, MissingError, digest_identity, identity, timestamp


SCHEMA = """
CREATE TABLE IF NOT EXISTS specimens (
    id TEXT PRIMARY KEY,
    version INTEGER NOT NULL,
    document TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS results (
    id TEXT PRIMARY KEY,
    tile_id TEXT NOT NULL REFERENCES specimens(id),
    kind TEXT NOT NULL,
    cache_key TEXT NOT NULL UNIQUE,
    document TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS results_tile ON results(tile_id, created_at DESC);
CREATE TABLE IF NOT EXISTS library (
    id TEXT PRIMARY KEY,
    tile_id TEXT NOT NULL REFERENCES specimens(id),
    document TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS library_source ON library(tile_id);
CREATE TABLE IF NOT EXISTS evidence (
    id TEXT PRIMARY KEY,
    tile_id TEXT NOT NULL REFERENCES specimens(id),
    document TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS evidence_tile ON evidence(tile_id, created_at DESC);
CREATE TABLE IF NOT EXISTS studies (
    id TEXT PRIMARY KEY,
    document TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS events (
    seq INTEGER PRIMARY KEY,
    tile_id TEXT NOT NULL REFERENCES specimens(id),
    action TEXT NOT NULL,
    message TEXT NOT NULL,
    recorded_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS events_tile ON events(tile_id, seq DESC);
CREATE TABLE IF NOT EXISTS received (
    id TEXT PRIMARY KEY,
    archive_digest TEXT NOT NULL UNIQUE,
    document TEXT NOT NULL,
    imported_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS metrology (
    id TEXT PRIMARY KEY,
    tile_id TEXT NOT NULL REFERENCES specimens(id),
    document TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS metrology_tile ON metrology(tile_id, created_at DESC);
CREATE TABLE IF NOT EXISTS seams (
    id TEXT PRIMARY KEY,
    document TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS settings (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


def make_id():
    return uuid4().hex[:24]


def document(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True, allow_nan=False)


def key_for(value):
    return hashlib.sha256(document(value).encode("utf8")).hexdigest()


def event(conn, tile_id, action, message):
    conn.execute(
        "INSERT INTO events(tile_id, action, message, recorded_at) VALUES (?, ?, ?, ?)",
        (tile_id, action, message, timestamp()),
    )


def artifact_row(row, label):
    if row is None:
        raise MissingError(f"{label} not found.")
    return json.loads(row["document"])


class BlobStore:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def path(self, digest):
        valid = digest_identity(digest)
        return self.directory / valid[:2] / valid

    def put(self, content):
        digest = hashlib.sha256(content).hexdigest()
        path = self.path(digest)
        if path.is_file():
            return digest
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.parent / f".{digest}.{make_id()}.tmp"
        try:
            with temp.open("xb") as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temp, path)
        finally:
            temp.unlink(missing_ok=True)
        return digest

    def require(self, digest):
        path = self.path(digest)
        if not path.is_file():
            raise MissingError("The image or artifact file could not be found.")
        return path

    def get(self, digest):
        return self.require(digest).read_bytes()


class Store:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.blobs = BlobStore(self.directory / "objects")
        self.database = self.directory / "weavedesk.sqlite"
        with self.connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.executescript(SCHEMA)
            conn.execute("PRAGMA user_version=1")

    @contextlib.contextmanager
    def connect(self):
        conn = sqlite3.connect(self.database, timeout=20, isolation_level=None)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA foreign_keys=ON")
            conn.execute("PRAGMA busy_timeout=20000")
            yield conn
        finally:
            conn.close()

    @contextlib.contextmanager
    def transaction(self, write=False):
        with self.connect() as conn:
            conn.execute("BEGIN IMMEDIATE" if write else "BEGIN")
            try:
                yield conn
            except BaseException:
                conn.rollback()
                raise
            else:
                conn.commit()

    def _tile(self, conn, identifier):
        tile_id = identity(identifier, "Specimen ID")
        row = conn.execute("SELECT document FROM specimens WHERE id=?", (tile_id,)).fetchone()
        return artifact_row(row, "Specimen")

    def tile(self, identifier):
        with self.connect() as conn:
            return self._tile(conn, identifier)

    def list_tiles(self):
        with self.connect() as conn:
            tiles = conn.execute(
                "SELECT document FROM specimens ORDER BY rowid DESC",
            ).fetchall()
        return [json.loads(row["document"]) for row in tiles]

    def _insert_tile(self, conn, fields, tool="Created specimen"):
        now = timestamp()
        tile = {
            **fields,
            "id": make_id(),
            "version": 1,
            "created_at": now,
            "updated_at": now,
            "photos": fields.get("photos", []),
            "regions": fields.get("regions", []),
            "plane": fields.get("plane"),
            "registrations": fields.get("registrations", {}),
            "calibration": fields.get("calibration"),
            "synthetic": fields.get("synthetic", False),
        }
        conn.execute(
            "INSERT INTO specimens(id, version, document, updated_at) VALUES (?, ?, ?, ?)",
            (tile["id"], tile["version"], document(tile), tile["updated_at"]),
        )
        event(conn, tile["id"], "create", tool)
        return tile

    def create_tile(self, fields):
        with self.transaction(write=True) as conn:
            return self._insert_tile(conn, fields)

    def edit_tile(self, identifier, expected_version, transform, action, message):
        with self.transaction(write=True) as conn:
            tile = self._tile(conn, identifier)
            self.require_version(tile, expected_version)
            transform(tile)
            tile["version"] += 1
            tile["updated_at"] = timestamp()
            conn.execute(
                "UPDATE specimens SET version=?, document=?, updated_at=? WHERE id=?",
                (tile["version"], document(tile), tile["updated_at"], tile["id"]),
            )
            event(conn, tile["id"], action, message)
            return tile

    @staticmethod
    def require_version(record, expected_version):
        if record["version"] != expected_version:
            suffix = f"Current version is {record['version']}."
            raise ConflictError(f"The record changed in another operation. Refresh and retry. {suffix}")

    def seed_tiles(self, source_key, seed_rows):
        with self.transaction(write=True) as conn:
            stored = conn.execute("SELECT value FROM settings WHERE key=?", (source_key,)).fetchone()
            if stored:
                return [self._tile(conn, specimen_id) for specimen_id in json.loads(stored["value"])]
            tiles = [self._insert_tile(conn, row, "Loaded synthetic training material") for row in seed_rows]
            conn.execute(
                "INSERT INTO settings(key,value) VALUES (?,?)",
                (source_key, document([tile["id"] for tile in tiles])),
            )
            return tiles

    def seeded_tiles(self, source_key):
        with self.transaction() as conn:
            stored = conn.execute("SELECT value FROM settings WHERE key=?", (source_key,)).fetchone()
            if stored is None:
                return []
            return [self._tile(conn, specimen_id) for specimen_id in json.loads(stored["value"])]

    def events(self, identifier, *, limit=35):
        with self.connect() as conn:
            source = self._tile(conn, identifier)
            rows = conn.execute(
                "SELECT seq, action, message, recorded_at FROM events WHERE tile_id=? ORDER BY seq DESC LIMIT ?",
                (source["id"], limit),
            )
            return [dict(row) for row in rows]

    def get_cached_result(self, cache_key):
        with self.connect() as conn:
            row = conn.execute("SELECT document FROM results WHERE cache_key=?", (cache_key,)).fetchone()
            return json.loads(row["document"]) if row else None

    def save_result(self, cache_key, fields):
        result = {
            **fields,
            "id": make_id(),
            "created_at": timestamp(),
            "decisions": {},
            "version": 1,
        }
        tile_id = result["tile_id"]
        with self.transaction(write=True) as conn:
            stored = conn.execute("SELECT document FROM results WHERE cache_key=?", (cache_key,)).fetchone()
            if stored is not None:
                return json.loads(stored["document"])
            conn.execute(
                "INSERT INTO results(id,tile_id,kind,cache_key,document,created_at) VALUES (?,?,?,?,?,?)",
                (result["id"], tile_id, result["kind"], cache_key, document(result), result["created_at"]),
            )
            event(conn, tile_id, "measure", result["label"])
        return result

    def _result(self, conn, identifier):
        result_id = identity(identifier, "Measurement ID")
        row = conn.execute("SELECT document FROM results WHERE id=?", (result_id,)).fetchone()
        return artifact_row(row, "Measurement")

    def result(self, identifier):
        with self.connect() as conn:
            return self._result(conn, identifier)

    def results(self, tile_id):
        source = identity(tile_id, "Specimen ID")
        with self.connect() as conn:
            self._tile(conn, source)
            rows = conn.execute(
                "SELECT document FROM results WHERE tile_id=? ORDER BY rowid DESC LIMIT 150", (source,),
            ).fetchall()
            return [json.loads(row["document"]) for row in rows]

    def save_decisions(self, identifier, expected_version, choices, picture, summary):
        with self.transaction(write=True) as conn:
            result = self._result(conn, identifier)
            self.require_version(result, expected_version)
            edited = 0
            for marker, decision in choices.items():
                old = result["decisions"].get(marker, {})
                if all(old.get(key) == decision[key] for key in ("decision", "notes")):
                    continue
                result["decisions"][marker] = {**decision, "updated_at": timestamp()}
                edited += 1
            if not edited:
                return result
            result["pictures"]["overlay"] = picture
            result["review_summary"] = summary
            result["version"] += 1
            conn.execute("UPDATE results SET document=? WHERE id=?", (document(result), result["id"]))
            event(conn, result["tile_id"], "review", f"Updated {edited} defect review decision(s)")
            return result

    def _reference(self, conn, identifier):
        chosen = identity(identifier, "Reference ID")
        row = conn.execute("SELECT document FROM library WHERE id=?", (chosen,)).fetchone()
        return artifact_row(row, "Reference")

    def reference(self, identifier):
        with self.connect() as conn:
            return self._reference(conn, identifier)

    def references(self, include_retired=False):
        with self.connect() as conn:
            refs = conn.execute("SELECT document FROM library ORDER BY rowid DESC LIMIT 250")
            values = [json.loads(row["document"]) for row in refs]
        return [item for item in values if include_retired or not item["retired"]]

    def add_reference(self, fields):
        now = timestamp()
        ref = {**fields, "id": make_id(), "retired": False, "version": 1, "created_at": now}
        with self.transaction(write=True) as conn:
            conn.execute(
                "INSERT INTO library(id,tile_id,document,updated_at) VALUES (?,?,?,?)",
                (ref["id"], ref["tile_id"], document(ref), now),
            )
            event(conn, ref["tile_id"], "reference", f"Approved visual reference: {ref['title']}")
        return ref

    def retire_reference(self, identifier, expected_version):
        with self.transaction(write=True) as conn:
            ref = self._reference(conn, identifier)
            self.require_version(ref, expected_version)
            if ref["retired"]:
                return ref
            ref["retired"] = True
            ref["version"] += 1
            ref["retired_at"] = timestamp()
            conn.execute(
                "UPDATE library SET document=?,updated_at=? WHERE id=?",
                (document(ref), ref["retired_at"], ref["id"]),
            )
            event(conn, ref["tile_id"], "reference", f"Retired visual reference: {ref['title']}")
            return ref

    def _evidence(self, conn, identifier):
        eid = identity(identifier, "Evidence ID")
        row = conn.execute("SELECT document FROM evidence WHERE id=?", (eid,)).fetchone()
        return artifact_row(row, "Evidence")

    def evidence(self, identifier):
        with self.connect() as conn:
            return self._evidence(conn, identifier)

    def list_evidence(self, identifier):
        chosen = identity(identifier, "Specimen ID")
        with self.connect() as conn:
            self._tile(conn, chosen)
            rows = conn.execute(
                "SELECT document FROM evidence WHERE tile_id=? ORDER BY rowid DESC LIMIT 100",
                (chosen,),
            )
            return [json.loads(row["document"]) for row in rows]

    def sources_for_evidence(self, identifier, result_ids):
        with self.transaction() as conn:
            tile = self._tile(conn, identifier)
            return tile, [self._result(conn, result_id) for result_id in result_ids]

    def save_evidence(self, fields):
        record = {**fields, "id": make_id(), "created_at": timestamp()}
        with self.transaction(write=True) as conn:
            conn.execute(
                "INSERT INTO evidence(id,tile_id,document,created_at) VALUES (?,?,?,?)",
                (record["id"], record["tile_id"], document(record), record["created_at"]),
            )
            event(conn, record["tile_id"], "evidence", f"Pinned inspection evidence: {record['title']}")
        return record

    def save_study(self, fields):
        study = {**fields, "id": make_id(), "created_at": timestamp()}
        with self.transaction(write=True) as conn:
            conn.execute(
                "INSERT INTO studies(id,document,updated_at) VALUES (?,?,?)",
                (study["id"], document(study), study["created_at"]),
            )
        return study

    def study(self, identifier):
        chosen = identity(identifier, "Study ID")
        with self.connect() as conn:
            row = conn.execute("SELECT document FROM studies WHERE id=?", (chosen,)).fetchone()
            return artifact_row(row, "Firing study")

    def studies(self):
        with self.connect() as conn:
            rows = conn.execute("SELECT document FROM studies ORDER BY rowid DESC LIMIT 50").fetchall()
        return [json.loads(row["document"]) for row in rows]

    def save_received(self, fields):
        packet = {**fields, "id": make_id(), "imported_at": timestamp()}
        with self.transaction(write=True) as conn:
            previous = conn.execute(
                "SELECT document FROM received WHERE archive_digest=?",
                (packet["archive_digest"],),
            ).fetchone()
            if previous is not None:
                return json.loads(previous["document"]), False
            conn.execute(
                "INSERT INTO received(id,archive_digest,document,imported_at) VALUES (?,?,?,?)",
                (packet["id"], packet["archive_digest"], document(packet), packet["imported_at"]),
            )
        return packet, True

    def received_by_digest(self, digest):
        with self.connect() as conn:
            row = conn.execute(
                "SELECT document FROM received WHERE archive_digest=?",
                (digest_identity(digest),),
            ).fetchone()
            return json.loads(row["document"]) if row else None

    def received(self, identifier):
        chosen = identity(identifier, "Received packet ID")
        with self.connect() as conn:
            row = conn.execute("SELECT document FROM received WHERE id=?", (chosen,)).fetchone()
            return artifact_row(row, "Received evidence packet")

    def received_packets(self):
        with self.connect() as conn:
            rows = conn.execute("SELECT document FROM received ORDER BY rowid DESC LIMIT 60").fetchall()
            return [json.loads(row["document"]) for row in rows]

    def save_metrology(self, fields):
        record = {**fields, "id": make_id(), "created_at": timestamp()}
        with self.transaction(write=True) as conn:
            conn.execute(
                "INSERT INTO metrology(id,tile_id,document,created_at) VALUES (?,?,?,?)",
                (record["id"], record["tile_id"], document(record), record["created_at"]),
            )
            event(conn, record["tile_id"], "geometry", f"Measured area in the tile plane: {record['title']}")
        return record

    def metrology(self, identifier):
        chosen = identity(identifier, "Geometry measurement ID")
        with self.connect() as conn:
            row = conn.execute("SELECT document FROM metrology WHERE id=?", (chosen,)).fetchone()
            return artifact_row(row, "Geometry measurement")

    def list_metrology(self, tile_id):
        chosen = identity(tile_id, "Specimen ID")
        with self.connect() as conn:
            self._tile(conn, chosen)
            rows = conn.execute(
                "SELECT document FROM metrology WHERE tile_id=? ORDER BY rowid DESC LIMIT 50", (chosen,),
            ).fetchall()
            return [json.loads(row["document"]) for row in rows]

    def save_seam(self, fields):
        record = {**fields, "id": make_id(), "created_at": timestamp(), "version": 1}
        with self.transaction(write=True) as conn:
            conn.execute(
                "INSERT INTO seams(id,document,created_at) VALUES(?,?,?)",
                (record["id"], document(record), record["created_at"]),
            )
        return record

    def _seam(self, conn, identifier):
        seam_id = identity(identifier, "Seam ID")
        row = conn.execute("SELECT document FROM seams WHERE id=?", (seam_id,)).fetchone()
        return artifact_row(row, "Ribbon seam")

    def seam(self, identifier):
        with self.connect() as conn:
            return self._seam(conn, identifier)

    def list_seams(self):
        with self.connect() as conn:
            rows = conn.execute("SELECT document FROM seams ORDER BY rowid DESC LIMIT 60").fetchall()
            return [json.loads(row["document"]) for row in rows]
