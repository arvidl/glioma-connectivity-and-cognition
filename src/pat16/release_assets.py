"""Standard-library, checksummed runtime bundles for local/future public releases."""
import gzip, hashlib, io, json, os, tarfile, tempfile
from pathlib import Path, PurePosixPath

CHUNK=1024*1024

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(CHUNK),b''):h.update(b)
    return h.hexdigest()

class SplitWriter:
    def __init__(self, directory, name, limit=900*1024**2):
        self.directory=Path(directory);self.directory.mkdir(parents=True,exist_ok=True)
        self.name=name;self.limit=limit;self.parts=[];self.file=None;self.size=0
    def write(self,b):
        total=len(b)
        while b:
            if self.file is None:
                p=self.directory/f'{self.name}.part{len(self.parts)+1:03d}'
                self.file=p.open('wb');self.parts.append(p);self.size=0
            n=min(len(b),self.limit-self.size);self.file.write(b[:n]);self.size+=n;b=b[n:]
            if self.size==self.limit:self.file.close();self.file=None
        return total
    def flush(self):
        if self.file:self.file.flush()
    def close(self):
        if self.file:self.file.close();self.file=None

class JoinedReader:
    def __init__(self,paths):self.paths=iter(paths);self.file=None
    def read(self,n):
        if n<0:raise ValueError('Streaming reader requires a bounded read')
        out=bytearray()
        while len(out)<n:
            if self.file is None:
                p=next(self.paths,None)
                if p is None:break
                self.file=Path(p).open('rb')
            b=self.file.read(n-len(out));out.extend(b)
            if not b:self.file.close();self.file=None
        return bytes(out)
    def close(self):
        if self.file:self.file.close()

def bundle(root,prefix,directory,name,limit=900*1024**2):
    root=Path(root);members=[];writer=SplitWriter(directory,name,limit)
    for p in sorted((root/prefix).rglob('*')):
        if p.is_symlink():raise ValueError(f'Symlink not allowed: {p}')
        if p.is_file() and not any(x in {'.DS_Store','__pycache__','.ipynb_checkpoints','.git','.cache'} for x in p.relative_to(root).parts):
            members.append(dict(path=p.relative_to(root).as_posix(),bytes=p.stat().st_size,sha256=digest(p)))
    try:
        with gzip.GzipFile(filename='',mode='wb',fileobj=writer,mtime=0,compresslevel=3) as gz:
            with tarfile.open(fileobj=gz,mode='w|',format=tarfile.PAX_FORMAT) as tar:
                for m in members:
                    info=tarfile.TarInfo(m['path']);info.size=m['bytes'];info.mode=0o644;info.mtime=0
                    with (root/m['path']).open('rb') as f:tar.addfile(info,f)
    finally:writer.close()
    return dict(prefix=prefix,members=members,parts=[dict(name=p.name,bytes=p.stat().st_size,sha256=digest(p)) for p in writer.parts])

def safe_path(root,name,prefix):
    pp=PurePosixPath(name)
    if pp.is_absolute() or '..' in pp.parts or not pp.parts or pp.parts[0]!=prefix or str(pp)!=name:
        raise ValueError(f'Unsafe asset member: {name}')
    root=Path(root).resolve();target=root.joinpath(*pp.parts)
    if any(p.is_symlink() for p in [target,*target.parents] if p!=root and root in p.parents):
        raise ValueError(f'Symlink destination: {name}')
    if root not in target.resolve().parents:raise ValueError(f'Outside destination: {name}')
    return target

def restore(root,spec,directory,overwrite=False):
    expected={m['path']:m for m in spec['members']}
    if len(expected)!=len(spec['members']):raise ValueError('Duplicate manifest member')
    for name in expected:safe_path(root,name,spec['prefix'])
    parts=[]
    for p in spec['parts']:
        if Path(p['name']).name!=p['name']:raise ValueError('Unsafe asset part name')
        path=Path(directory)/p['name']
        if not path.is_file() or path.stat().st_size!=p['bytes'] or digest(path)!=p['sha256']:
            raise ValueError(f'Asset checksum/size mismatch: {path.name}')
        parts.append(path)
    source=JoinedReader(parts);seen=set()
    try:
        with tarfile.open(fileobj=source,mode='r|gz') as tar:
            for info in tar:
                if not info.isfile() or info.name not in expected or info.name in seen:
                    raise ValueError(f'Unexpected, duplicate or nonregular member: {info.name}')
                seen.add(info.name);m=expected[info.name];dest=safe_path(root,info.name,spec['prefix'])
                if info.size!=m['bytes']:raise ValueError('Member size mismatch')
                same=dest.is_file() and dest.stat().st_size==m['bytes'] and digest(dest)==m['sha256']
                if dest.exists() and not same and not overwrite:raise ValueError(f'Modified destination; use --overwrite explicitly: {info.name}')
                dest.parent.mkdir(parents=True,exist_ok=True);temp=None;h=hashlib.sha256();size=0
                try:
                    if not same:
                        fd,temp=tempfile.mkstemp(prefix='.asset-',dir=dest.parent);out=os.fdopen(fd,'wb')
                    else:out=None
                    try:
                        f=tar.extractfile(info)
                        for b in iter(lambda:f.read(CHUNK),b''):
                            size+=len(b);h.update(b)
                            if out:out.write(b)
                    finally:
                        if out:out.close()
                    if size!=m['bytes'] or h.hexdigest()!=m['sha256']:raise ValueError('Member checksum mismatch')
                    if temp:os.replace(temp,dest);temp=None
                finally:
                    if temp:Path(temp).unlink(missing_ok=True)
    finally:source.close()
    if seen!=set(expected):raise ValueError('Missing archive members')

def verify(root,spec):
    failures=[]
    for m in spec['members']:
        p=safe_path(root,m['path'],spec['prefix'])
        if not p.is_file() or p.stat().st_size!=m['bytes'] or digest(p)!=m['sha256']:failures.append(m['path'])
    if failures:raise ValueError(f'{len(failures)} missing/changed files: '+', '.join(failures[:5]))
