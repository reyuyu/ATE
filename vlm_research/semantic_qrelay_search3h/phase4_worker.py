from . import common as c
c.P=c.PHASE1/'phase4_policy'
c.PRIVATE=c.PRIVATE.parent/'semantic_policy_phase4'
from .worker import worker
if __name__=='__main__':worker(0)
