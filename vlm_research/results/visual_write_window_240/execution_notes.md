# Execution notes

Source base e139c76e32edd3eacdf44d7172069bc37b230b60; frozen execution source 3b1950f9780676bac874d0a97b53341ab8c57ec7. Independent worktree and branch; historical outputs and main untouched.

Historical Handoff B: 20/240 question overlap, older LM-head shape path; no audited same-240 same-path All-Writer records. Native Full also differs in answer image access. No main record reused.

CPU 31 tests passed. Same-shape torch.equal HF/Native/no-op checks and 18 conditions at each of the three frozen smoke questions passed. No-Writer image independence: all 40 frozen pairs, every effective logit and complete token sequence strictly equal. Control performed before main matrix.

One model on only physical GPU2 (UUID in protocol). Each condition starts with an empty independent cache; all 32 layers audited at every forward. No o_proj ablation, skipped samples, retry, precision change, tolerance relaxation or source edits during GPU execution.

Completed 4320 main generations + 80 independent swap endpoints, plus 78 necessary smoke generations (3 questions x [HF + Native + 6 no-op + 18 active conditions]). All are counted separately. GPU process exited before CPU reporting. GPU0/1 processes were left untouched; no background wait or polling job was created.

Raw first-token arrays live in the server resource directory, recorded by path/SHA256 in checkpoints; large arrays are not committed to GitHub.
