# Execution notes

Base commit 6b9467d6fb4e0f8535040809bee60397ca798ea8. Frozen source commit 0ef85f2ca07115641c76945e414607965b316a71. Independent worktree/branch; main and all historical artifacts untouched.

41 CPU tests passed. A pre-freeze formatting-only trailing whitespace cleanup had identical Python AST; the earlier CPU record was preserved and the final source hashes were revalidated before committing. No source edit occurred after freezing.

Audited four historical frontier baselines: 960 records, saved first-logit arrays, masks, cache traces, tokens, scores and exact input bindings. Native Full 240 records audited separately; its old paired R records were also checked by the historical audit helper, but neither were pooled into the frontier.

Three fixed real 7B GPU Smoke questions passed, each with all six new conditions, all six no-ops, four historical conditions through both engines, HF native and custom native. 66 necessary smoke generations, separate from the 1440 main generations. All effective same-shape logits were torch.equal; historical saved baseline first vectors and full token sequences matched.

Only physical GPU0, selected and rechecked idle by UUID. One model, serial independent real caches. No o_proj ablation, skipped samples, retries, precision changes or tolerance relaxation. GPU process exited before CPU analysis. Other GPU/process activity was left untouched. No background GPU-wait or polling job.

All 1440 new main answers completed. Reused main baseline records 960, separate native Full reference 240. First-token raw arrays stay in the server resource folder, referenced by path and SHA256; GitHub does not contain these large arrays.
