"""Fetch original official ZIP members through curl's system-CA TLS transport.

Used on the local transfer host when the compute-server route returns HTTP 503.
The frozen ZIP central directory supplies CRC and exact byte lengths.
"""
import concurrent.futures,hashlib,json,pathlib,struct,subprocess,sys,time,zlib

def main():
    index=json.loads(pathlib.Path(sys.argv[1]).read_text());out=pathlib.Path(sys.argv[2]);out.mkdir(parents=True,exist_ok=True)
    def fetch(a,b,key):
        data=out/(key+'.range');headers=out/(key+'.headers')
        for attempt in range(4):
            p=subprocess.run(['curl.exe','--silent','--show-error','--fail','--location','--max-time','45',
                '--range',f'{a}-{b}','--dump-header',str(headers),'--output',str(data),
                index['url']+f'?transfer_range={a}-{b}&attempt={attempt}'],capture_output=True,
                creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            if p.returncode==0:break
            if attempt==3:raise RuntimeError(p.stderr.decode(errors='replace'))
            time.sleep(2*(attempt+1))
        h=headers.read_text().lower();body=data.read_bytes()
        assert f'content-range: bytes {a}-{b}/{index["size"]}' in h,h
        assert ('etag: '+index['etag']).lower() in h,h
        assert len(body)==b-a+1
        data.unlink();headers.unlink();return body
    def one(x):
        q=x['image_id'];p=out/(q+'.jpg')
        if p.exists():b=p.read_bytes()
        else:
            a=x['header_offset'];header=fetch(a,a+29,q+'-header');assert header[:4]==b'PK\x03\x04'
            n,e=struct.unpack_from('<HH',header,26);a+=30+n+e
            comp=fetch(a,a+x['compress_size']-1,q+'-body')
            b=comp if x['compress_type']==0 else zlib.decompress(comp,-15)
            p.write_bytes(b)
        assert len(b)==x['file_size'] and zlib.crc32(b)&0xffffffff==x['CRC']
        return dict(image_id=q,sha256=hashlib.sha256(b).hexdigest(),bytes=len(b),zip_crc32=x['CRC'])
    records=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        for r in pool.map(one,index['members']):
            records.append(r)
            if len(records)%20==0:print('ORIGINAL_IMAGES',len(records),flush=True)
    (out/'local_transfer_audit.json').write_text(json.dumps(dict(source=index['url'],etag=index['etag'],images=records),indent=2))
    print('COMPLETE',len(records),flush=True)

if __name__=='__main__':main()
