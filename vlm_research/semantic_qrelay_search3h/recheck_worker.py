from . import common as c
c.P=c.PHASE1/'phase8_fixed_recheck';c.PRIVATE=c.PRIVATE.parent/'semantic_fixed_recheck'
from .worker import worker
if __name__=='__main__':worker(0)
