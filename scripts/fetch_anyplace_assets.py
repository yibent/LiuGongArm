"""Fetch selected members of official public AnyPlace ZIPs via HTTP Range.

Avoids downloading the 35 GB training set or all task-specific checkpoints.
zipfile verifies CRC; only explicitly selected archive members are extracted.
"""
import argparse
import io
from pathlib import Path
import time
import zipfile
import requests

SIZES={'anyplace_ckpts.zip':371478276,'anyplace_eval.zip':806182145}


class RemoteZip(io.RawIOBase):
    def __init__(self,name):
        self.url=f'https://huggingface.co/datasets/yuchiallanzhao/anyplace/resolve/main/{name}'
        self.size=SIZES[name];self.position=0;self.cached_start=-1;self.cached=b''
        self.session=requests.Session()
    def seekable(self): return True
    def tell(self): return self.position
    def seek(self,offset,whence=0):
        self.position=offset+(self.position if whence==1 else self.size if whence==2 else 0)
        if self.position<0: raise ValueError('Negative seek')
        return self.position
    def read(self,size=-1):
        end=self.size if size<0 else min(self.size,self.position+size)
        pieces=[]
        while self.position<end:
            if not self.cached_start<=self.position<self.cached_start+len(self.cached):
                start=self.position;stop=min(self.size,start+4*1024*1024)-1
                for attempt in range(5):
                    try:
                        response=self.session.get(self.url,headers={'Range':f'bytes={start}-{stop}'},timeout=(15,45))
                        response.raise_for_status()
                        if response.status_code!=206 or response.headers.get('Content-Range')!=f'bytes {start}-{stop}/{self.size}':
                            raise ValueError('Server did not honor exact byte range')
                        if len(response.content)!=stop-start+1: raise IOError('Incomplete range')
                        self.cached_start=start;self.cached=response.content
                        print(f'Fetched bytes {start}-{stop}',flush=True)
                        break
                    except (requests.RequestException,IOError,ValueError):
                        if attempt==4: raise
                        time.sleep(1)
            count=min(end-self.position,self.cached_start+len(self.cached)-self.position)
            offset=self.position-self.cached_start
            pieces.append(self.cached[offset:offset+count]);self.position+=count
        return b''.join(pieces)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive',choices=list(SIZES))
    parser.add_argument('--member',action='append',default=[])
    parser.add_argument('--output',type=Path,default=Path('_models/anyplace'))
    args=parser.parse_args()
    with zipfile.ZipFile(RemoteZip(args.archive)) as archive:
        if not args.member:
            for entry in archive.infolist():print(f'{entry.file_size}\t{entry.filename}')
            return
        root=args.output.resolve();root.mkdir(parents=True,exist_ok=True)
        for name in args.member:
            destination=(root/name).resolve()
            if not destination.is_relative_to(root):raise ValueError('Unsafe archive path')
            if destination.exists():raise FileExistsError(destination)
            archive.extract(name,root)
            print(f'Extracted {destination}',flush=True)


if __name__=='__main__':main()
