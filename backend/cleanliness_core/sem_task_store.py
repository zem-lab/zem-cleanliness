"""Durable incremental SEM state; task.json remains a periodic full checkpoint.

Each SQLite transaction is committed before the next device action. Checkpoints
carry an epoch and sequence; replay ignores already included transactions. Old
epochs are retained so concurrent readers of an older checkpoint stay valid.
Copy the entire task directory (including SQLite and its WAL), not task.json
alone, while a task is active. Export/report readers must use load_sem_task.
"""
import json
import sqlite3
import uuid
import hashlib
from collections import Counter
from contextlib import closing
from pathlib import Path

from .atomic_json import write_json_atomic
from . import sem_progress

DB_NAME = 'sem_state.sqlite3'


def load_sem_task(directory):
    directory = Path(directory).resolve()
    manifest = json.loads((directory/'task.json').read_text(encoding='utf-8'))
    marker = manifest.get('incrementalState')
    if not marker:
        return manifest
    db = directory/DB_NAME
    if not db.is_file():
        raise ValueError('SEM 增量记录缺失，拒绝以旧快照恢复任务')
    with closing(sqlite3.connect(db.as_uri()+'?mode=ro', uri=True, timeout=15)) as connection:
        rows = connection.execute('SELECT seq,payload FROM changes WHERE epoch=? AND seq>? ORDER BY seq',
                                  (marker['epoch'], marker['seq'])).fetchall()
    particles = {p['sequence']:p for p in manifest['particles']}
    tiles = {t['id']:t for t in manifest['plan']['tiles']}
    seq = marker['seq']
    for number, raw in rows:
        if number != seq+1:
            raise ValueError('SEM 增量记录序号不连续，拒绝自动恢复')
        patch = json.loads(raw)
        if patch['taskId'] != manifest['taskId']:
            raise ValueError('SEM 增量记录任务不匹配')
        manifest.update(patch['header'])
        if patch['tile'] is not None:
            tiles[patch['tile']['id']].clear()
            tiles[patch['tile']['id']].update(patch['tile'])
        for particle in patch['particles']:
            particles[particle['sequence']] = particle
        timing = patch['timing']
        if timing is not None:
            index = timing['index']
            runs = manifest.setdefault('timingRuns', [])
            if index == len(runs):
                runs.append({'events':[]})
            if index >= len(runs) or len(runs[index]['events']) != timing['eventOffset']:
                raise ValueError('SEM 计时增量不连续')
            runs[index].update(timing['header'])
            runs[index]['events'].extend(timing['events'])
        seq = number
    manifest['particles'] = [particles[k] for k in sorted(particles)]
    manifest['incrementalState'] = {'epoch':marker['epoch'], 'seq':seq}
    return manifest


class SemTaskStore:
    def __init__(self, directory, manifest):
        self.directory = Path(directory).resolve()
        self.connection = sqlite3.connect(self.directory/DB_NAME, timeout=15)
        self.connection.execute('PRAGMA journal_mode=WAL')
        self.connection.execute('PRAGMA synchronous=FULL')
        self.connection.execute('CREATE TABLE IF NOT EXISTS changes (epoch TEXT, seq INTEGER, payload TEXT NOT NULL, PRIMARY KEY(epoch,seq))')
        self.connection.commit()
        sem_progress.initialize(self.connection)
        self.epoch, self.seq = uuid.uuid4().hex, 0
        self.event_count = len((manifest.get('timingRuns') or [{}])[-1].get('events', []))
        self.particles = {p['sequence']:self.digest(self.encode(p)) for p in manifest['particles']}
        self.statuses = {p['sequence']:p['status'] for p in manifest['particles']}
        self.counts = Counter(self.statuses.values())
        self.tile_statuses = {t['id']:t['status'] for t in manifest['plan']['tiles']}
        self.completed = sum(v=='COMPLETED' for v in self.tile_statuses.values())
        manifest['incrementalState'] = {'epoch':self.epoch, 'seq':0}
        try:
            self.checkpoint(manifest)
            with self.connection:
                self.connection.execute('DELETE FROM ui_tiles')
                self.connection.execute('DELETE FROM ui_particles')
                self.connection.executemany('INSERT INTO ui_tiles VALUES (?,?,?)',
                    [(t['id'],0,self.encode(sem_progress.tile_view(t))) for t in manifest['plan']['tiles']])
                self.connection.executemany('INSERT INTO ui_particles VALUES (?,?)',
                    [(p['sequence'],self.encode(sem_progress.particle_view(p))) for p in manifest['particles']])
                self.connection.execute('INSERT OR REPLACE INTO ui_header VALUES (1,?,?,?)',
                    (self.epoch,0,self.encode(sem_progress.header_view(manifest,self.counts,self.completed))))
        except BaseException:
            self.close()
            raise

    @staticmethod
    def encode(value):
        return json.dumps(value, ensure_ascii=False, separators=(',',':'), allow_nan=False)

    @staticmethod
    def digest(text):
        return hashlib.sha256(text.encode('utf-8')).digest()

    def save(self, manifest, tile=None, *, dirty_particles=None):
        changed, encoded = [], {}
        # Only serialize the current field, never all prior spectra/contours.
        for particle in manifest['particles'] if dirty_particles is None else dirty_particles:
            if dirty_particles is None and tile is not None and particle['tileId'] != tile['id']:
                continue
            text = self.encode(particle)
            digest = self.digest(text)
            if self.particles.get(particle['sequence']) != digest:
                changed.append(particle)
                encoded[particle['sequence']] = digest
        counts = self.counts.copy()
        for p in changed:
            old_status=self.statuses.get(p['sequence'])
            if old_status is not None: counts[old_status]-=1
            counts[p['status']]+=1
        completed = self.completed
        if tile is not None:
            completed += int(tile['status']=='COMPLETED')-int(self.tile_statuses[tile['id']]=='COMPLETED')
        manifest['input']['selectedCount']=counts.get('SUCCEEDED',0)+counts.get('MEASURED',0)
        runs = manifest.get('timingRuns', [])
        timing = None
        if runs:
            run = runs[-1]
            timing = {'index':len(runs)-1, 'header':{k:v for k,v in run.items() if k!='events'},
                      'eventOffset':self.event_count, 'events':run['events'][self.event_count:]}
        patch = {'taskId':manifest['taskId'], 'header':{k:v for k,v in manifest.items()
                 if k not in {'particles','plan','timingRuns','incrementalState'}},
                 'tile':tile, 'particles':changed, 'timing':timing}
        payload = self.encode(patch)
        with self.connection:
            self.connection.execute('INSERT INTO changes VALUES (?,?,?)',(self.epoch,self.seq+1,payload))
            self.connection.execute('INSERT OR REPLACE INTO ui_header VALUES (1,?,?,?)',
                (self.epoch,self.seq+1,self.encode(sem_progress.header_view(manifest,counts,completed))))
            if tile is not None:
                self.connection.execute('INSERT OR REPLACE INTO ui_tiles VALUES (?,?,?)',
                    (tile['id'],self.seq+1,self.encode(sem_progress.tile_view(tile))))
            self.connection.executemany('INSERT OR REPLACE INTO ui_particles VALUES (?,?)',
                [(p['sequence'],self.encode(sem_progress.particle_view(p))) for p in changed])
        # Advance only after a successful durable commit.
        self.seq += 1
        self.particles.update(encoded)
        self.counts,self.completed = counts,completed
        self.statuses.update({p['sequence']:p['status'] for p in changed})
        if tile is not None: self.tile_statuses[tile['id']]=tile['status']
        if runs:
            self.event_count = len(runs[-1]['events'])
        manifest['incrementalState'] = {'epoch':self.epoch,'seq':self.seq}

    def checkpoint(self, manifest):
        write_json_atomic(self.directory/'task.json', manifest, indent=None)

    def close(self):
        self.connection.close()
