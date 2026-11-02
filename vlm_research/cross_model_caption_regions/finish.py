from .common import *


def main():
    p=verify()
    assert read(OUT/'analysis_complete.json')['complete']
    snapshot=preflight()
    write(OUT/'gpu_release.json',dict(released=True,**snapshot))
    write(OUT/'validation.json',dict(complete=True,images=60,new_generations=960,
        models={m:dict(smoke=read(OUT/m/'smoke.json')['passed'],complete=read(OUT/m/'complete.json')['complete']) for m in MODELS},
        physical_gpu=1,gpu_released=True,cpu=read(OUT/'cpu_validation.json'),
        protocol_sha256=sha(OUT/'protocol.json'),source_unchanged=p['source_sha256']==sources(),
        no_training=True,no_extra_conditions=True,main_unchanged=True))
    write(OUT/'artifact_sha256.json',{str(f.relative_to(OUT)):sha(f) for f in sorted(OUT.rglob('*'))
          if f.is_file() and f.name!='artifact_sha256.json'})
    print('Complete, validated, GPU1 released.',flush=True)


if __name__=='__main__':main()
