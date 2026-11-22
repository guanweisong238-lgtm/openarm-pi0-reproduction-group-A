"""Offline action-window indexing and timing; does not control robot hardware."""
from dataclasses import dataclass
import math
import numpy as np

@dataclass(frozen=True)
class ReplanRequest:
 last_executed_index:int
 known_prefix:np.ndarray
 chunk_start_index:int


def make_replan_request(window,*,chunk_start_index=0,executed_actions=14):
 """After 14 actions, reuse the last action and next six as a seven-action prefix."""
 w=np.asarray(window)
 if w.shape!=(50,16) or not np.isfinite(w).all():raise ValueError('window must be finite (50,16)')
 if chunk_start_index not in (0,7) or executed_actions!=14:raise ValueError('unsupported timing')
 i=chunk_start_index+executed_actions-1
 if i+7>len(w):raise ValueError('not enough actions to form prefix')
 return ReplanRequest(last_executed_index=i,known_prefix=w[i:i+7].copy(),chunk_start_index=chunk_start_index)


def select_after_inference(old_window,new_window,request,*,completed_actions):
 """Select timestamp-aligned new action after old step 20; no smoothing or interpolation.

 completed_actions counts actions executed within the current 20-action chunk; it may exceed 20
 if inference is late and the old window remains available. On-time inference must still wait
 until action 20. If the model output does not preserve the known prefix, fail loudly.
 """
 old=np.asarray(old_window);new=np.asarray(new_window)
 if old.shape!=(50,16) or new.shape!=(50,16) or not np.isfinite(old).all() or not np.isfinite(new).all():raise ValueError('invalid windows')
 if completed_actions<14:raise ValueError('inference cannot be complete before request begins')
 if not np.allclose(new[:7],request.known_prefix,rtol=0,atol=1e-6):raise ValueError('new model did not preserve seven known actions')
 completed=max(20,completed_actions)
 if request.chunk_start_index+completed>=len(old):raise ValueError('old plan exhausted; explicit fallback is required')
 new_index=7+(completed-20)
 if new_index>=len(new):raise ValueError('new plan expired; explicit fallback is required')
 return {'switch_after_action':completed,'old_index':request.chunk_start_index+completed,'new_index':new_index,'new_action':new[new_index].copy(),'extra_old_steps':completed-20}


def completed_actions_on_next_boundary(latency_seconds,*,hz=30):
 if latency_seconds<0 or hz<=0:raise ValueError('invalid time')
 return 14+math.ceil(latency_seconds*hz-1e-12)
