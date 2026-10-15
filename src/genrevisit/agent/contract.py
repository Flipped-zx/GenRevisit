"""Agent input validation, action policy and message rendering."""
from __future__ import annotations
import hashlib
import math
from typing import Any
from .schema_loader import validate_instance
from .json_canonical import canonical_json
from .skills import DEFAULT_SKILL_MANIFEST

POLICY_ID = 'joint_gm_sft@1'

INPUT_VERSION = 'joint_planner_input@1'

EPS = 1e-9

SYSTEM_PROMPT = '''You are the Joint image retry planner. Emit exactly one canonical Action Protocol v0.5 JSON: query_skill, generate_image, edit_image, or submit_attempt. No prose, backend, scores, or paths in your output.
Your objective is the highest observed Geneval2 GM. At equal GM prefer more passed constraints, then the earlier attempt. control.gm_best_attempt_id is authoritative for submission. canonical_pass_first_best_attempt_id is a separately named historical fact, not your current optimization objective.
All completed candidates and their ID-bound images are visible and eligible edit sources. Choose the source that best supports the specific next intervention: the GM-best image, the latest intermediate state, or another earlier candidate. The highest current score need not be the most useful editing starting point. There is no preference quota for any source class. Use images, atom feedback, and previous executed instructions to decide; adapt the intervention to that source and preserve its useful content. Avoid blindly repeating a failed intervention.
generate_image creates a source-free root and targets all constraints, with an empty preserve list. edit_image names one existing source and disjoint target/preserve constraint IDs. Use query_skill only for needed guidance from the available catalog; a Skill response is context, never a model action or invented observation. Never repeat an unchanged retrieved Skill. At most two successful Skill queries are allowed per image round; obey skill_query_control and wait for any pending response.
Submit the existing GM-best. If no image budget remains, reason_code is best_available_under_budget. Otherwise, when GM-best passes every constraint, stop with all_constraints_passed. Otherwise you may stop with no_productive_action_remaining. You may not generate or edit after a required stop. Never predict or invent evaluator outcomes.'''

def digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()

def control(state: dict) -> dict:
    order, attempts = state['attempt_order'], state['attempts']
    if len(order) != len(set(order)) or set(order) != set(attempts):
        raise ValueError('candidate identity mismatch')
    if state['task_spec']['max_image_attempts'] != 5 or state['remaining_budget'] != 5-len(order):
        raise ValueError('Joint requires five total images with canonical remaining budget')
    if len(order)>5 or state['latest_attempt_id'] != (order[-1] if order else None):
        raise ValueError('invalid completed candidate order')
    for i, aid in enumerate(order):
        a=attempts[aid];score=a['primary_score']
        if isinstance(score,bool) or not isinstance(score,(int,float)) or not math.isfinite(score) or not 0<=score<=1:
            raise ValueError('missing finite candidate GM')
        if a['parent_attempt_id'] is not None and a['parent_attempt_id'] not in order[:i]:
            raise ValueError('source must precede its child')
    gm=max(order,key=lambda aid:(attempts[aid]['primary_score'],len(attempts[aid]['passed_constraint_ids']),-order.index(aid))) if order else None
    pbest=max(order,key=lambda aid:(len(attempts[aid]['passed_constraint_ids']),attempts[aid]['primary_score'],-order.index(aid))) if order else None
    if pbest!=state['best_attempt_id']:
        raise ValueError('canonical P-best mismatch')
    value=dict(schema_version='0.2',policy_id='gm_fresh_control@1',arm='joint',
        gm_best_attempt_id=gm,gm_best_score=attempts[gm]['primary_score'] if gm else None,
        canonical_pass_first_best_attempt_id=pbest,latest_attempt_id=order[-1] if order else None,
        editable_source_ids=list(order),visible_candidate_ids=list(order),submittable_candidate_ids=[gm] if gm else [],
        tie_break='higher_pass_count_then_earlier',remaining_image_budget=state['remaining_budget'])
    validate_instance(value,'gm_continuation_control_v0_2.schema.json')
    return value

def required_reason(state: dict) -> str | None:
    c=control(state);best=c['gm_best_attempt_id']
    if best is None:return None
    if c['remaining_image_budget']==0:return 'best_available_under_budget'
    if len(state['attempts'][best]['passed_constraint_ids'])==len(state['task_spec']['constraints']):return 'all_constraints_passed'
    return None

def validate_action(action: dict, state: dict, active_skill_ids=(), skill_query_control=None) -> None:
    validate_instance(action,'action_protocol_v0_5.schema.json')
    c=control(state);kind=action['action'];args=action['arguments']
    known={x['constraint_id'] for x in state['task_spec']['constraints']}
    targets=set(args.get('target_constraint_ids',[]));preserve=set(args.get('preserve_constraint_ids',[]))
    if targets&preserve or not (targets|preserve)<=known:raise ValueError('invalid target/preserve IDs')
    stop=required_reason(state)
    if stop and kind!='submit_attempt':raise ValueError('required stop: '+stop)
    if kind=='submit_attempt':
        if not c['submittable_candidate_ids'] or args['selected_attempt_id']!=c['gm_best_attempt_id']:
            raise ValueError('submit must select actual GM-best')
        if args['reason_code']!=(stop or 'no_productive_action_remaining'):raise ValueError('invalid submit reason')
    elif kind=='edit_image':
        if args['source_attempt_id']not in c['editable_source_ids']:raise ValueError('source not completed')
    elif kind=='generate_image':
        if targets!=known or preserve:raise ValueError('generate must target entire task without source preservation')
    elif kind=='query_skill':
        ids=set(args['skill_ids']);catalog={s['skill_id']for s in DEFAULT_SKILL_MANIFEST}
        if not ids<=catalog or ids&set(active_skill_ids):raise ValueError('unknown or already active Skill')
        if skill_query_control is not None:
            if skill_query_control['pending_query_ids']:raise ValueError('pending Skill query response')
            if skill_query_control['remaining_queries']<=0:raise ValueError('two Skill queries per image round exhausted')

def skill_control(events: list[dict]) -> dict:
    start=0
    for i,e in enumerate(events):
        if e['event_type']=='geneval2_completed':start=i+1
    current=events[start:]
    queries=[e['event_id']for e in current if e['event_type']=='action_validated'and e['payload']['action']['action']=='query_skill']
    returned={e['payload']['query_action_event_id']for e in current if e['event_type']=='skill_returned'}
    return dict(max_queries=2,completed_queries=len(returned),remaining_queries=max(0,2-len(returned)),pending_query_ids=[q for q in queries if q not in returned])

def validate_input(value: dict) -> None:
    validate_instance(value,'joint_planner_input_v0_1.schema.json')
    ids=[c['attempt_id']for c in value['candidates']]
    if ids!=value['control']['visible_candidate_ids']:raise ValueError('image candidate order mismatch')
    if len({c['image_artifact_id']for c in value['candidates']})!=len(ids):raise ValueError('duplicate image binding')
    known={x['constraint_id']for x in value['task_context']['atom_constraints']}
    for c in value['candidates']:
        if {r['constraint_id']for r in c['constraint_results']}!=known:raise ValueError('incomplete atom feedback')
        if len(c['constraint_results'])!=len(known):raise ValueError('duplicate atom feedback')
        if sorted(c['passed_constraint_ids'])!=sorted(r['constraint_id']for r in c['constraint_results']if r['status']=='pass'):
            raise ValueError('pass IDs disagree with atom feedback')
    if value['available_skill_catalog']!=DEFAULT_SKILL_MANIFEST:raise ValueError('Joint Skill catalog drift')
    sq=value['skill_query_control']
    if sq['remaining_queries']!=max(0,2-sq['completed_queries']):raise ValueError('inconsistent Skill query budget')
    if control(state_from_input(value))!=value['control']:raise ValueError('input control differs from candidate facts')

def state_from_input(value: dict) -> dict:
    """Reconstruct comparator/reference facts, not a replacement canonical memory."""
    ids=[c['attempt_id']for c in value['candidates']]
    task=dict(max_image_attempts=value['task_context']['max_image_attempts'],constraints=value['task_context']['atom_constraints'])
    attempts={c['attempt_id']:dict(primary_score=c['gm'],passed_constraint_ids=c['passed_constraint_ids'],parent_attempt_id=c['parent_attempt_id'])for c in value['candidates']}
    return dict(attempt_order=ids,attempts=attempts,task_spec=task,remaining_budget=value['control']['remaining_image_budget'],
        latest_attempt_id=value['control']['latest_attempt_id'],best_attempt_id=value['control']['canonical_pass_first_best_attempt_id'])

def render_messages(packet: dict, image_refs: list[dict], target_action: dict | None = None) -> list[dict]:
    validate_input(packet)
    expected=[(c['attempt_id'],c['image_artifact_id'])for c in packet['candidates']]
    if [(r['attempt_id'],r['artifact_id'])for r in image_refs]!=expected:raise ValueError('images not in canonical candidate order')
    messages=[dict(role='system',content=SYSTEM_PROMPT,loss_weight=0,token_source='joint_system'),
        dict(role='user',content=canonical_json(packet),image_refs=image_refs,loss_weight=0,token_source='joint_prefix_input')]
    if target_action is not None:
        validate_instance(target_action,'action_protocol_v0_5.schema.json')
        messages.append(dict(role='assistant',content=canonical_json(target_action),loss_weight=1,token_source='canonical_action'))
    return messages
