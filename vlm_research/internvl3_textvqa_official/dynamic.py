"""Isolated mechanical dynamic-length port of the frozen Late9 physical engine.

Only legacy total-image-length constants change. No historical module globals,
permissions, layer sets, precision or kernel calculations are patched in place.
The port is not GPU-validated until the explicit Smoke gate has passed.
"""
import ast
from functools import lru_cache
import sys
from types import ModuleType

from .common import HISTORY, sha


class DynamicLength(ast.NodeTransformer):
    def __init__(self):
        self.functions = []
        self.changes = []

    def visit_FunctionDef(self, node):
        self.functions.append(node)
        result = self.generic_visit(node)
        self.functions.pop()
        return result

    def visit_Constant(self, node):
        if not isinstance(node.value, int) or node.value != 256:
            return node
        function = self.functions[-1]
        params = {a.arg for a in function.args.args}
        if 'self' in params:
            expression = 'len(self.roles["image"])'
        elif function.name == 'layout':
            expression = 'len(roles["image"])'
        elif function.name in ['generate', 'gpu_ready']:
            expression = 'len(ready["roles"]["image"])' if function.name == 'generate' else 'len(roles["image"])'
        else:
            raise AssertionError(('UNREVIEWED_IMAGE_LENGTH_CONSTANT', function.name, node.lineno))
        self.changes.append(dict(line=node.lineno, function=function.name, old=256, replacement=expression))
        return ast.copy_location(ast.parse(expression, mode='eval').body, node)


def adapted(path, name, base=None):
    before = sha(path)
    tree = ast.parse(path.read_text())
    if base is not None:
        tree.body = [n for n in tree.body if not (isinstance(n, ast.ImportFrom) and
            n.module == 'vlm_research.internvl3_svh_physical_d_pilot')]
    transform = DynamicLength()
    tree = ast.fix_missing_locations(transform.visit(tree))
    module = ModuleType(name)
    module.__file__ = str(path)
    module.__dict__['base'] = base
    sys.modules[name] = module
    exec(compile(tree, str(path), 'exec'), module.__dict__)
    assert sha(path) == before
    return module, dict(path=str(path), sha256=before, replacements=transform.changes)


@lru_cache(maxsize=1)
def engines():
    if str(HISTORY) not in sys.path:
        sys.path.append(str(HISTORY))
    base, a = adapted(HISTORY/'vlm_research/internvl3_svh_physical_d_pilot/engine.py',
        'ate_internvl_dynamic_base')
    late, b = adapted(HISTORY/'vlm_research/internvl3_svh_r_terminal_windows_textvqa/engine.py',
        'ate_internvl_dynamic_late9', base)
    assert late.SKIP == base.SKIP == set(range(2, 6))
    return base, late, [a, b]
