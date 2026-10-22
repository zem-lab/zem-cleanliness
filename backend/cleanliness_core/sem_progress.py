"""Small, durable UI projection, separate from scientific/recovery data."""
import json
import sqlite3
from collections import Counter
from contextlib import closing
from pathlib import Path

PARTICLE_FIELDS=('sequence','tileId','status','lengthUm','areaUm2','reason')
TILE_FIELDS=('id','x','y','status','preview','error')


def particle_view(p):
    result={k:p[k] for k in PARTICLE_FIELDS if k in p}
    if p.get('spectrum'):
        result['spectrum']={'elements':sorted(p['spectrum'].get('elements',[]),key=lambda e:-e['value'])[:3]}
    return result


def tile_view(t):
    return {k:t[k] for k in TILE_FIELDS if k in t}


def header_view(manifest,counts,completed):
    result={k:manifest[k] for k in ('taskId','mode','status','input','navigation','request','model','reports',
            'reportState','reportError','cleanup','postCompletionGunShutdown','error','updatedAt','createdAt','completedAt') if k in manifest}
    result['plan']={k:v for k,v in manifest['plan'].items() if k!='tiles'}
    result['stats']={'completedTiles':completed,'particleCount':sum(counts.values()),
        'measured':counts.get('SUCCEEDED',0)+counts.get('MEASURED',0),'rejected':counts.get('REJECTED',0)}
    return result


def initialize(connection):
    connection.executescript('''
    CREATE TABLE IF NOT EXISTS ui_header (id INTEGER PRIMARY KEY, epoch TEXT, seq INTEGER, payload TEXT);
    CREATE TABLE IF NOT EXISTS ui_tiles (id INTEGER PRIMARY KEY, seq INTEGER, payload TEXT);
    CREATE TABLE IF NOT EXISTS ui_particles (id INTEGER PRIMARY KEY, payload TEXT);
    ''')


def read_progress(directory,cursor=''):
    directory=Path(directory).resolve()
    db=directory/'sem_state.sqlite3'
    if not db.is_file(): return None
    with closing(sqlite3.connect(db.as_uri()+'?mode=ro',uri=True,timeout=10)) as conn:
        conn.execute('BEGIN')  # Header, tiles and recent particles from one version.
        if not conn.execute("SELECT 1 FROM sqlite_master WHERE name='ui_header'").fetchone(): return None
        row=conn.execute('SELECT epoch,seq,payload FROM ui_header WHERE id=1').fetchone()
        if not row: return None
        epoch,seq,payload=row
        new_cursor=f'{epoch}:{seq}'
        if cursor==new_cursor: return {'cursor':new_cursor,'unchanged':True}
        try:
            old_epoch,old_seq=cursor.split(':'); old_seq=int(old_seq)
            reset=old_epoch!=epoch or old_seq<0 or old_seq>seq
        except (ValueError,AttributeError):
            reset=True; old_seq=-1
        tiles=[json.loads(r[0]) for r in conn.execute('SELECT payload FROM ui_tiles WHERE seq>? ORDER BY id',(-1 if reset else old_seq,))]
        particles=[json.loads(r[0]) for r in conn.execute('SELECT payload FROM ui_particles ORDER BY id DESC LIMIT 100')][::-1]
        task=json.loads(payload)
        task['particles']=particles
        task['taskDirectory']=str(directory)
        return {'cursor':new_cursor,'reset':reset,'task':task,'tiles':tiles}


def read_summary(directory):
    db=Path(directory).resolve()/'sem_state.sqlite3'
    if not db.is_file(): return None
    with closing(sqlite3.connect(db.as_uri()+'?mode=ro',uri=True,timeout=10)) as conn:
        if not conn.execute("SELECT 1 FROM sqlite_master WHERE name='ui_header'").fetchone():return None
        row=conn.execute('SELECT payload FROM ui_header WHERE id=1').fetchone()
        if not row:return None
        return json.loads(row[0])


def refresh_inactive_header(directory,manifest):
    """Publish manually exported report links without replaying large UI data."""
    db=Path(directory).resolve()/'sem_state.sqlite3'
    if not db.is_file(): return
    with closing(sqlite3.connect(db,timeout=10)) as conn:
        if not conn.execute("SELECT 1 FROM sqlite_master WHERE name='ui_header'").fetchone():return
        row=conn.execute('SELECT epoch,seq FROM ui_header WHERE id=1').fetchone()
        if not row:return
        counts=Counter(p['status'] for p in manifest['particles'])
        completed=sum(t['status']=='COMPLETED' for t in manifest['plan']['tiles'])
        payload=json.dumps(header_view(manifest,counts,completed),ensure_ascii=False,separators=(',',':'))
        with conn:
            conn.execute('UPDATE ui_header SET seq=?,payload=? WHERE id=1',(row[1]+1,payload))
