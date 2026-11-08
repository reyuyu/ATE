# Frozen eight-method E2E@32 execution sources

This publication contains the exact execution source subset and independent HF 4.49 compatibility code used by the unified RTX 4090 run. Complete upstream source trees and archives remain on the persistent work disk and in the Windows source-archive backup; GitHub omits bulky upstream figures/assets and archives.

Full upstream file manifests and retrieval receipts describe those complete mirrors. They do not imply that every upstream asset is vendored in this GitHub directory. Full archive commits and SHA256 values are also recorded in `../results/unified_e2e32_4090/official_source_packages.json`.

- FastV: d1659729b5bf1be225e99ee15783deeea80f63b1, static KV K3-192.
- SparseVLM ICML v1.5: 734d9003879689713a132b83297dfad060a5c9af; nominal192 schedule, actual layer mean206.5 after official recycling.
- V2Drop: 2a9431150f8ace3a1d9abcdc96a7d46acca5597df; frozen historical 576->288->144->1 point.
- ApET: 70cc01b215f4d74c5661c699919fe3aeda48bed0, 288->96, basis10.
- DUET: cd69e489f39a2a51e3b278a470dc8dc32c00e0fb, official C+all 307->153->0; not the paper main-table C variant.
- Native HF: actual imported Transformers4.49.0 source snapshots.
- SVH/SVH-R: byte-identical frozen ATE core snapshots under `ate_svh/frozen_sources`; original repository algorithm files are untouched.

All numerical results, environment paths, per-layer cache audit, compatibility details and limitations are in the linked result directory. Installing or importing this source capsule does not start an experiment. The original runner still requires the frozen model, images and its historical source evidence; it is not a self-contained data or model release.
