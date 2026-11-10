"""Single-owner WAL checkpoints. No prediction replacement on resume."""
import fcntl
import hashlib
import json
import sqlite3
import time
import zlib


class Store:
    def __init__(self, path, protocol_hash):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = path.with_suffix('.lock').open('a')
        fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.db = sqlite3.connect(path, timeout=60)
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.execute('PRAGMA synchronous=FULL')
        self.db.executescript('''
            CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS predictions (
                key TEXT NOT NULL, condition TEXT NOT NULL, state TEXT NOT NULL,
                payload BLOB, payload_sha256 TEXT, updated REAL NOT NULL,
                PRIMARY KEY(key, condition));
            CREATE TABLE IF NOT EXISTS events (time REAL, kind TEXT, payload TEXT);
        ''')
        self.db.execute('INSERT OR IGNORE INTO meta VALUES (?,?)', ('protocol', protocol_hash))
        assert self.db.execute('SELECT value FROM meta WHERE key="protocol"').fetchone()[0] == protocol_hash
        self.db.execute('UPDATE predictions SET state="PENDING" WHERE state="RUNNING"')
        self.db.commit()

    def plan(self, keys, conditions):
        with self.db:
            self.db.executemany('INSERT OR IGNORE INTO predictions VALUES (?,?,"PENDING",NULL,NULL,?)',
                [(k, c, time.time()) for k in keys for c in conditions])

    def state(self, key, condition):
        return self.db.execute('SELECT state FROM predictions WHERE key=? AND condition=?', (key, condition)).fetchone()[0]

    def begin(self, key, condition):
        assert self.state(key, condition) == 'PENDING'
        with self.db:
            self.db.execute('UPDATE predictions SET state="RUNNING",updated=? WHERE key=? AND condition=?', (time.time(), key, condition))

    def finish(self, key, condition, record):
        assert record['status'] in ('COMPLETE', 'SKIPPED_OOM', 'BLOCKED')
        assert self.state(key, condition) == 'RUNNING'
        raw = json.dumps(record, sort_keys=True, ensure_ascii=True).encode()
        with self.db:
            self.db.execute('UPDATE predictions SET state=?,payload=?,payload_sha256=?,updated=? WHERE key=? AND condition=?',
                (record['status'], zlib.compress(raw), hashlib.sha256(raw).hexdigest(), time.time(), key, condition))

    def event(self, kind, payload):
        with self.db:
            self.db.execute('INSERT INTO events VALUES (?,?,?)', (time.time(), kind, json.dumps(payload, sort_keys=True)))

    def counts(self):
        return dict(self.db.execute('SELECT state,count(*) FROM predictions GROUP BY state'))

    def records(self):
        for key, condition, state, blob, digest in self.db.execute('SELECT key,condition,state,payload,payload_sha256 FROM predictions ORDER BY key,condition'):
            if blob is None:
                continue
            raw = zlib.decompress(blob)
            assert hashlib.sha256(raw).hexdigest() == digest
            value = json.loads(raw)
            assert (value['key'], value['condition'], value['status']) == (key, condition, state)
            yield value

    def close(self):
        self.db.execute('PRAGMA wal_checkpoint(TRUNCATE)')
        self.db.close()
        self.lock.close()
