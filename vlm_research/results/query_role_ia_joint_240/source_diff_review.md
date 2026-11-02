# Source review before final delivery

Historical modules are unchanged. The joint engine is an isolated copy of the frozen historical role generator; the sole computational change is the explicit joint position union. The generator still starts each call with history=None, calls model with logits_to_keep=1, creates a fresh real cache, and runs unchanged MLP/residual/mask/KV code.

The new orchestration reuses only immutable Full/I/A results. Smoke verifies all-step exact parity with the old generator and first-logit/token equality with saved server arrays. Three saved-baseline comparisons per smoke must pass before the one-condition collection loop. The collection loop does not generate new Full/I/A or R answers.

Analysis replays all joint token/text/score/first-logit and cache/hook records, revalidates historical raw arrays and files, and computes all contrasts from one bootstrap draw matrix. Joint-specific harm is Full/I/A all correct with joint wrong. No R hidden equality requirement is introduced.

Engine difference:

```diff
--- historical_engine.py
+++ joint_engine.py
@@ -31,6 +31,8 @@
 def target_positions(role, layout, decode):
     if role not in CONDITIONS:
         raise ValueError('Unknown frozen condition')
+    if role == 'I+A-Bypass':
+        return [0] if decode else sorted(layout['I']+layout['A'])
     if role == 'Full':
         return []
     if decode:
```
