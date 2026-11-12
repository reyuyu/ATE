from pathlib import Path
import ast,json,shutil
A=Path('/home/featurize/ate-experiments/svh-cross-opt-20261113');B=A/'opt_b'
def replace_method(p,cls,name,body):
 s=p.read_text();ls=s.splitlines(True);t=ast.parse(s)
 c=next(n for n in t.body if isinstance(n,ast.ClassDef) and n.name==cls)
 n=next(n for n in c.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name==name)
 indent=' '*n.col_offset
 text='\n'.join(indent+l if l else '' for l in body.strip('\n').splitlines())+'\n'
 ls[n.lineno-1:n.end_lineno]=[text];p.write_text(''.join(ls))
def append_methods(p,cls,body):
 s=p.read_text();ls=s.splitlines(True);c=next(n for n in ast.parse(s).body if isinstance(n,ast.ClassDef) and n.name==cls)
 ls.insert(c.end_lineno,'\n'+'\n'.join('    '+l if l else '' for l in body.strip('\n').splitlines())+'\n');p.write_text(''.join(ls))
