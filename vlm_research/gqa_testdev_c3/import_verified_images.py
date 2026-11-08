"""Import mirror JPEG bytes only after matching the official ZIP member CRC/size."""
import hashlib,json,pathlib,zlib

def main():
    import pyarrow.parquet as pq
    from PIL import Image
    p=pathlib.Path('/data/lk/ATE-resources/data/gqa-testdev-balanced-official')
    index=json.loads((p/'official_zip_members.json').read_text());members={x['image_id']:x for x in index['members']}
    parquet=p/'hf_testdev_images.parquet';file_sha=hashlib.sha256(parquet.read_bytes()).hexdigest()
    assert file_sha=='ba0a57a72c483d6e4a390ecbbee05d07a55365aeac3e93f4a923ac61c9e465eb'
    rows=pq.read_table(parquet).to_pylist();assert len(rows)==len(members)==398
    audit=[];direct=0
    for r in rows:
        x=members[r['id']];b=r['image']['bytes'];assert len(b)==x['file_size'] and (zlib.crc32(b)&0xffffffff)==x['CRC']
        target=p/'images'/f'{r["id"]}.jpg'
        matched=target.exists()
        if matched:assert target.read_bytes()==b;direct+=1
        else:target.write_bytes(b)
        with Image.open(target) as im:im.verify()
        audit.append(dict(image_id=r['id'],path=str(target),sha256=hashlib.sha256(b).hexdigest(),bytes=len(b),
            zip_member=x['filename'],zip_crc32=x['CRC'],direct_stanford_bytes_also_verified=matched))
    q=p/'testdev_balanced_questions.json';raw=json.loads(q.read_text());assert len(raw)==12578
    record=dict(split='testdev_balanced',questions=12578,question_source='https://downloads.cs.stanford.edu/nlp/data/gqa/questions1.2.zip',
        question_member=q.name,question_file_sha256=hashlib.sha256(q.read_bytes()).hexdigest(),question_archive_etag='"5c9a1875-59531234"',
        image_source=index['url'],image_archive_etag=index['etag'],images=audit,all_zip_crc_verified=True,
        transport_source='https://hf-mirror.com/datasets/lmms-lab-encoder/GQA/resolve/main/testdev_balanced_images/testdev-00000-of-00001.parquet',
        upstream_repository='https://huggingface.co/datasets/lmms-lab-encoder/GQA',transport_sha256=file_sha,
        equivalence='All 398 JPEG byte lengths and CRCs match official ZIP members; direct original byte equality additionally checked for existing downloads.',
        direct_original_sha_crosschecks=direct,limitation='Remaining JPEGs were not independently re-downloaded from Stanford for a second SHA comparison.')
    (p/'download_audit.json').write_text(json.dumps(record,indent=2)+'\n');print('VERIFIED',len(audit),'DIRECT_BYTE_CROSSCHECKS',direct)

if __name__=='__main__':main()
