from . import common as c
c.P=c.PHASE1/'phase6_fixed_compositions';c.PRIVATE=c.PRIVATE.parent/'semantic_fixed_compositions'
from .worker import worker
if __name__=='__main__':worker(0)
