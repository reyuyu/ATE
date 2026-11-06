"""Extract only required official ZIP members over validated HTTP byte ranges."""
import io,json,pathlib,urllib.request,zipfile,hashlib,concurrent.futures,shutil
DEST=pathlib.Path('/data/lk/ATE-resources/data/gqa-testdev-balanced-official')
BASE='https://downloads.cs.stanford.edu/nlp/data/gqa/'

class Remote(io.RawIOBase):
    def __init__(self,url):
        self.url=url;self.pos=0;self.etag=None;self.cache=[]
        data,h=self.fetch(0,31);self.size=int(h['Content-Range'].split('/')[-1]);self.etag=h['ETag']
    def fetch(self,a,b):
        if b-a+1>65536:
            spans=[(start,min(b,start+65535)) for start in range(a,b+1,65536)]
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                parts=list(pool.map(lambda span:self.fetch(*span),spans))
            return b''.join(p[0] for p in parts),parts[-1][1]
        import time
        cache=DEST/'range_cache';cache.mkdir(parents=True,exist_ok=True)
        key=cache/(hashlib.sha256(self.url.encode()).hexdigest()[:12]+f'-{a}-{b}')
        meta=key.with_suffix('.json')
        if key.exists() and meta.exists():
            headers=json.loads(meta.read_text());data=key.read_bytes()
            assert len(data)==b-a+1 and h(data)==headers['sha256']
            if self.etag:assert headers['ETag']==self.etag
            return data,headers
        for attempt in range(8):
            # A distinct query avoids a cached origin 503 for another byte range.
            target=self.url if attempt%2==0 else self.url.replace('https:','http:')
            req=urllib.request.Request(target+f'?ate_range={a}-{b}&retry={attempt}',headers={'Range':f'bytes={a}-{b}'})
            try:r=urllib.request.urlopen(req,timeout=45);break
            except Exception as exc:
                print('RANGE_RETRY',a,b,attempt,str(exc),flush=True)
                if attempt==7:raise
                time.sleep(2*(attempt+1))
        with r:
            assert r.status==206 and r.headers['Content-Range'].startswith(f'bytes {a}-{b}/')
            if self.etag:assert r.headers['ETag']==self.etag
            data=r.read();assert len(data)==b-a+1
            headers=dict(r.headers);headers['sha256']=h(data)
            key.write_bytes(data);meta.write_text(json.dumps(headers))
            return data,headers
    def seek(self,n,w=0):
        self.pos=n if w==0 else self.pos+n if w==1 else self.size+n
        return self.pos
    def tell(self):return self.pos
    def seekable(self):return True
    def read(self,n=-1):
        if n<0:n=self.size-self.pos
        n=min(n,self.size-self.pos)
        if not n:return b''
        a=self.pos;b=a+n
        for start,data in self.cache:
            if start<=a and b<=start+len(data):self.pos=b;return data[a-start:b-start]
        data,_=self.fetch(a,b-1);self.pos=b
        if n<30*2**20:self.cache.append((a,data))
        return data

def h(data):return hashlib.sha256(data).hexdigest()

def main():
    DEST.mkdir(parents=True,exist_ok=True)
    q=Remote(BASE+'questions1.2.zip');z=zipfile.ZipFile(q)
    names=z.namelist();print('TESTDEV MEMBERS',[n for n in names if 'testdev' in n],flush=True)
    name=next(n for n in names if pathlib.Path(n).name=='testdev_balanced_questions.json')
    data=z.read(name);(DEST/'testdev_balanced_questions.json').write_bytes(data)
    raw=json.loads(data);assert len(raw)==12578 and len(set(raw))==12578
    assert all(x.get('answer') is not None and x['isBalanced'] for x in raw.values())
    ids=sorted({x['imageId'] for x in raw.values()});print('QUESTIONS',len(raw),'IMAGES',len(ids),flush=True)
    r=Remote(BASE+'images.zip');images=zipfile.ZipFile(r);members={pathlib.Path(x.filename).stem:x for x in images.infolist() if x.filename.endswith('.jpg')}
    assert all(i in members for i in ids)
    (DEST/'images').mkdir(exist_ok=True);audit=[]
    def one(i):
        assert shutil.disk_usage('/data').free>20*2**30
        info=members[i];remote=Remote(BASE+'images.zip')
        # Reuse the validated central directory descriptor; no full archive download.
        with zipfile.ZipFile(remote) as archive:
            archive.NameToInfo={info.filename:info};b=archive.read(info.filename)
        p=DEST/'images'/f'{i}.jpg';p.write_bytes(b)
        from PIL import Image
        with Image.open(p) as im:im.verify()
        return dict(image_id=i,path=str(p),sha256=h(b),bytes=len(b),zip_member=info.filename,zip_crc32=info.CRC)
    # One central-directory fetch per worker is intentionally avoided below by
    # using a minimal ZipFile object with the frozen original ZipInfo offsets.
    def fast(i):
        import struct,zlib
        assert shutil.disk_usage('/data').free>20*2**30
        info=members[i];a=info.header_offset
        p=DEST/'images'/f'{i}.jpg'
        if p.exists():
            b=p.read_bytes();assert len(b)==info.file_size and zlib.crc32(b)&0xffffffff==info.CRC
            from PIL import Image
            with Image.open(p) as im:im.verify()
            return dict(image_id=i,path=str(p),sha256=h(b),bytes=len(b),zip_member=info.filename,zip_crc32=info.CRC)
        header,_=r.fetch(a,a+29);assert header[:4]==b'PK\x03\x04'
        n,e=struct.unpack_from('<HH',header,26);offset=a+30+n+e
        comp,_=r.fetch(offset,offset+info.compress_size-1)
        b=comp if info.compress_type==0 else zlib.decompress(comp,-15)
        assert len(b)==info.file_size and zlib.crc32(b)&0xffffffff==info.CRC
        p=DEST/'images'/f'{i}.jpg'
        if p.exists():assert p.read_bytes()==b
        else:p.write_bytes(b)
        from PIL import Image
        with Image.open(p) as im:im.verify()
        return dict(image_id=i,path=str(p),sha256=h(b),bytes=len(b),zip_member=info.filename,zip_crc32=info.CRC)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        for result in pool.map(fast,ids):
            audit.append(result)
            if len(audit)%100==0:print('OFFICIAL_IMAGES',len(audit),flush=True)
    record=dict(question_source=BASE+'questions1.2.zip',question_member=name,question_file_sha256=h(data),question_archive_etag=q.etag,
        image_source=BASE+'images.zip',image_archive_etag=r.etag,questions=len(raw),images=audit,split='testdev_balanced',all_zip_crc_verified=True)
    (DEST/'download_audit.json').write_text(json.dumps(record,indent=2)+'\n');print('DOWNLOAD COMPLETE',flush=True)

if __name__=='__main__':main()
