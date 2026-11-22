"""Check 14/20 indexing and late-inference alignment without any hardware."""
import numpy as np
from real_action_window import make_replan_request,select_after_inference,completed_actions_on_next_boundary
old=np.repeat(np.arange(50)[:,None],16,axis=1).astype(float);req=make_replan_request(old)
np.testing.assert_array_equal(req.known_prefix[:,0],np.arange(13,20))
new=np.repeat(np.arange(13,63)[:,None],16,axis=1).astype(float)
for finished,expected_index,expected_extra in [(14,7,0),(20,7,0),(23,10,3)]:
 choice=select_after_inference(old,new,req,completed_actions=finished)
 assert choice['new_index']==expected_index and choice['extra_old_steps']==expected_extra
 assert choice['new_action'][0]==old[choice['old_index'],0]
assert completed_actions_on_next_boundary(.2)==20 and completed_actions_on_next_boundary(.275)==23
# The second and subsequent chunks begin after the seven known prefix values.
req2=make_replan_request(new,chunk_start_index=7)
np.testing.assert_array_equal(req2.known_prefix[:,0],np.arange(33,40))
try:select_after_inference(old,np.zeros((50,16)),req,completed_actions=20)
except ValueError:pass
else:raise AssertionError('mismatched prefix was accepted')
print('PASS: 14/20 timing, 7-action prefix, timestamp-aligned late switching, subsequent chunk, invalid prefix rejection')
