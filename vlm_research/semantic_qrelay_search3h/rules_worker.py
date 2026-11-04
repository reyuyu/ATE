from . import common as c
c.P=c.PHASE1/'phase5_fixed_rules';c.PRIVATE=c.PRIVATE.parent/'semantic_fixed_rules'
from .worker import worker
if __name__=='__main__':worker(0)
