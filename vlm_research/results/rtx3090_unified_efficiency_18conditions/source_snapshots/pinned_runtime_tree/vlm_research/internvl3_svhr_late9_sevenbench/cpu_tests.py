"""Reuse the authority's Float64 physical/dense regression, limited to Late9."""
import os
import tempfile
from pathlib import Path
from .common import OUT, KEEP, read, write


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    from vlm_research.internvl3_svh_r_terminal_windows_textvqa import cpu_tests as frozen
    with tempfile.TemporaryDirectory(prefix="late9-cpu-", dir=OUT) as tmp:
        (Path(tmp) / "results").mkdir()
        os.environ["ATE_DEEP_RUN"] = tmp
        frozen.CONFIGS = {"Late9_L19_27": (KEEP, 28)}
        frozen.main()
        write(OUT / "cpu_validation.json", read(Path(tmp) / "results/cpu_tests.json"))


if __name__ == "__main__":
    main()
