from . import common as c
c.P=c.PHASE1/'phase7_fixed_refinements';c.PRIVATE=c.PRIVATE.parent/'semantic_fixed_refinements'
from .worker import worker
if __name__=='__main__':worker(0)
