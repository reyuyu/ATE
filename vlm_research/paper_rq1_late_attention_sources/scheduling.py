"""Four frozen data shards scheduled on three explicitly authorized GPUs."""
from .protocol import GPUS

SHARDS = tuple(range(4))
# Keep each image's native generation and interventions on the same physical GPU.
SHARD_GPU = {0: 1, 1: 2, 2: 3, 3: 1}


def queues():
    return {gpu: tuple(rank for rank in SHARDS if SHARD_GPU[rank] == gpu)
            for gpu in sorted(set(SHARD_GPU.values()))}


def configuration():
    return dict(version=2, authorized_physical_gpus=[1, 2, 3],
                logical_shards=4, images_per_shard=50,
                shard_physical_gpu={str(rank): SHARD_GPU[rank] for rank in SHARDS},
                shard_gpu_uuid={str(rank): GPUS[SHARD_GPU[rank]] for rank in SHARDS},
                same_image_full_ablation_same_gpu=True,
                rationale='User authorized three GPUs; frozen samples and scientific conditions unchanged. GPU0 excluded.')
