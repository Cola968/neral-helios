"""Versioned single-owner chat sync. SQLite transactions, bounded data, tombstones."""

import json,re,sqlite3,time

import storage_codec

from pathlib import Path



class Conflict(Exception):pass



class ChatStore:

    def __init__(self,path):

        self.path=Path(path);self.path.parent.mkdir(parents=True,exist_ok=True)

        with self.connect() as db:

            db.execute('PRAGMA journal_mode=WAL')

            db.execute('CREATE TABLE IF NOT EXISTS chats(id TEXT PRIMARY KEY, revision INTEGER NOT NULL, updated REAL NOT NULL, deleted INTEGER NOT NULL, body TEXT NOT NULL)')

    def connect(self):return sqlite3.connect(self.path,timeout=10)

    def all(self):

        with self.connect() as db:

            return [dict(id=r[0],revision=r[1],updated=r[2],deleted=bool(r[3]),chat=storage_codec.loads(r[4])) for r in db.execute('SELECT * FROM chats ORDER BY updated DESC LIMIT 500')]

    def put(self,key,revision,chat=None,deleted=False):

        if not isinstance(key,str) or not re.fullmatch(r'[a-zA-Z0-9-]{16,64}',key):raise ValueError('Ungültige Chat-ID.')

        if type(revision) is not int or revision<0 or type(deleted) is not bool:raise ValueError('Ungültige Revision.')

        if not deleted:

            if not isinstance(chat,dict) or chat.get('temporary'):raise ValueError('Temporäre Chats bleiben lokal.')

            if not isinstance(chat.get('title'),str) or len(chat['title'])>100:raise ValueError('Ungültiger Titel.')

            messages=chat.get('messages')

            if not isinstance(messages,list) or len(messages)>200:raise ValueError('Maximal 200 Nachrichten je synchronisiertem Chat.')

            clean=[]

            for m in messages:

                if not isinstance(m,dict) or m.get('role') not in ['user','assistant'] or not isinstance(m.get('content'),str) or len(m['content'])>32768:raise ValueError('Ungültige Nachricht.')

                clean.append({k:m[k] for k in ['role','content','mode','checkpoint','meta','interrupted','source','imported'] if k in m and isinstance(m[k],(str,bool))})

            draft=chat.get('draft','')

            if not isinstance(draft,str) or len(draft)>8192:raise ValueError('Entwurf zu lang.')

            chat={'id':key,'title':chat['title'],'messages':clean,'draft':draft,'mode':chat.get('mode','chat') if chat.get('mode') in ['chat','solve','code'] else 'chat','imported':bool(chat.get('imported')),'temporary':False}

        body=json.dumps(chat or {},ensure_ascii=True)

        if len(body.encode())>240000:raise ValueError('Chat zu groß für die Synchronisierung; bitte exportieren.')

        with self.connect() as db:

            db.execute('BEGIN IMMEDIATE')

            row=db.execute('SELECT revision FROM chats WHERE id=?',(key,)).fetchone()

            if (row[0] if row else 0)!=revision:raise Conflict('Der Chat wurde auf einem anderen Gerät geändert.')

            if not row and db.execute('SELECT COUNT(*) FROM chats').fetchone()[0]>=500:raise ValueError('Maximal 500 synchronisierte Chats inklusive Löschmarkierungen.')

            next_revision=revision+1

            db.execute('INSERT OR REPLACE INTO chats VALUES(?,?,?,?,?)',(key,next_revision,time.time(),int(deleted),storage_codec.dumps(chat or {},ensure_ascii=True)))

            return next_revision

