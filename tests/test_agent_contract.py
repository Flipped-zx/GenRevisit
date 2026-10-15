import json
from pathlib import Path
import pytest
from genrevisit.agent import contract as joint
FIX=Path(__file__).parent/"fixtures/agent"

def state():
    attempts={f'a_{i:03d}':dict(primary_score=gm,passed_constraint_ids=passed,parent_attempt_id=None)
        for i,(gm,passed)in enumerate([(.7,['c_001']),(.4,['c_001','c_002']),(.5,['c_002'])])}
    return dict(attempt_order=list(attempts),attempts=attempts,remaining_budget=2,latest_attempt_id='a_002',best_attempt_id='a_001',
        task_spec=dict(max_image_attempts=5,constraints=[{'constraint_id':f'c_{i:03d}'}for i in range(1,4)]))

def edit(src):
    return dict(schema_version='0.5',action='edit_image',arguments=dict(source_attempt_id=src,target_constraint_ids=['c_003'],preserve_constraint_ids=[],instruction='Add a clearly visible red balloon.'))

@pytest.mark.parametrize('src',['a_000','a_001','a_002'])
def test_all_three_sources_are_legal(src):
    joint.validate_action(edit(src),state())

def test_gm_pbest_and_tie_break():
    s=state();assert joint.control(s)['gm_best_attempt_id']=='a_000'
    s['attempts']['a_002']['primary_score']=.7
    assert joint.control(s)['gm_best_attempt_id']=='a_000'
    s['attempts']['a_002']['passed_constraint_ids']=['c_001','c_002'];s['best_attempt_id']='a_002'
    assert joint.control(s)['gm_best_attempt_id']=='a_002'

def test_submit_gm_best_with_fewer_passes_is_positive_legal():
    action=dict(schema_version='0.5',action='submit_attempt',arguments=dict(selected_attempt_id='a_000',reason_code='no_productive_action_remaining'))
    joint.validate_action(action,state())
    action['arguments']['selected_attempt_id']='a_001'
    with pytest.raises(ValueError,match='GM-best'):joint.validate_action(action,state())

def test_input_control_tamper_rejected():
    packet=json.loads((FIX/'empty_input.json').read_text())
    packet['control']['remaining_image_budget']=4
    with pytest.raises(Exception):joint.validate_input(packet)

def test_skill_budget_and_pending_are_shared_by_runtime():
    s=state()
    query=dict(schema_version='0.5',action='query_skill',arguments=dict(skill_ids=['object_identity_presence'],target_constraint_ids=['c_003']))
    exhausted=dict(max_queries=2,completed_queries=2,remaining_queries=0,pending_query_ids=[])
    pending=dict(max_queries=2,completed_queries=0,remaining_queries=2,pending_query_ids=['evt_pending'])
    for value in [exhausted,pending]:
        with pytest.raises(ValueError):joint.validate_action(query,s,[],value)
    events=[dict(event_id='q1',event_type='action_validated',payload={'action':query}),
            dict(event_id='r1',event_type='skill_returned',payload={'query_action_event_id':'q1'}),
            dict(event_id='q2',event_type='action_validated',payload={'action':query})]
    assert joint.skill_control(events)['pending_query_ids']==['q2']
    events.append(dict(event_type='geneval2_completed',payload={}))
    assert joint.skill_control(events)==dict(max_queries=2,completed_queries=0,remaining_queries=2,pending_query_ids=[])


def test_rendering_binds_images_and_rejects_reordering():
    import copy
    packet = json.loads((FIX / 'empty_input.json').read_text())
    constraint_ids = [c['constraint_id'] for c in packet['task_context']['atom_constraints']]
    action = dict(schema_version='0.5', action='generate_image', arguments=dict(
        target_constraint_ids=constraint_ids, preserve_constraint_ids=[],
        instruction='Generate the requested image.'))
    packet['candidates'] = [dict(
        attempt_id=f'a_{i:03d}', image_artifact_id=f'img_{i:03d}',
        parent_attempt_id=None, gm=gm, passed_constraint_ids=[],
        constraint_results=[dict(constraint_id=cid, status='fail') for cid in constraint_ids],
        executed_action=copy.deepcopy(action)) for i, gm in enumerate([0.6, 0.4])]
    state = joint.state_from_input(packet)
    state.update(remaining_budget=3, latest_attempt_id='a_001', best_attempt_id='a_000')
    packet['control'] = joint.control(state)
    refs = [dict(attempt_id=c['attempt_id'], artifact_id=c['image_artifact_id'],
                 uri=f"images/{c['image_artifact_id']}.png") for c in packet['candidates']]
    messages = joint.render_messages(packet, refs)
    assert json.loads(messages[1]['content']) == packet
    assert messages[1]['image_refs'] == refs
    with pytest.raises(ValueError, match='order'):
        joint.render_messages(packet, list(reversed(refs)))
    packet['control']['gm_best_attempt_id'] = 'a_001'
    with pytest.raises(ValueError, match='control'):
        joint.validate_input(packet)
