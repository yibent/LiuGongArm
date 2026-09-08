"""Small, observable failure taxonomy shared with the task supervisor."""
class PlacementSpaceUnavailable(RuntimeError):
    """Observed placement needs a new region/view/arrangement before pickup."""


class LocalizationFailure(RuntimeError):
    """Select/reobserve an instance before asking either motion backend to grasp."""


class RegraspRequired(RuntimeError):
    """The held grasp cannot satisfy the goal; another pose provider is insufficient."""


def failure_feedback(error, phase, holding, evaluation=None):
    message = str(error)
    if isinstance(error, RegraspRequired):
        return {'code': 'REGRASP_REQUIRED', 'phase': phase, 'holding_verified': bool(holding.get('verified')),
            'suggested_recovery': ['preserve_grasp', 'stage_and_regrasp'], 'message': message}
    orientation = (evaluation or {}).get('postconditions',{}).get('requested_orientation',{})
    if orientation and not orientation.get('satisfied'):
        return {'code':'WRONG_ORIENTATION','phase':phase,'holding_verified':bool(holding.get('verified')),
            'suggested_recovery':['inspect_actual_endpoints','regrasp_with_orientation_constraint'],
            'message':message,'measured_postcondition':orientation}
    check = (evaluation or {}).get('postconditions',{}).get('requested_cell',{})
    if check and not check.get('satisfied'):
        code = 'WRONG_CELL' if check.get('minimum_wall_margin_m',0)<-.002 else 'NOT_SEATED'
        recovery = ['inspect_actual_placement','regrasp_with_divider_clearance','preserve_requested_cell']
        if holding.get('verified'): recovery = ['preserve_grasp','place_held']+recovery
        return {'code':code,'phase':phase,'holding_verified':bool(holding.get('verified')),
            'suggested_recovery':recovery,'message':message,'measured_postcondition':check}
    relation = (evaluation or {}).get('postconditions',{}).get('requested_relation',{})
    if relation and not relation.get('satisfied'):
        return {'code':'NOT_SEATED','phase':phase,'holding_verified':bool(holding.get('verified')),
            'suggested_recovery':['inspect_contact_feature','replan_contact_approach'],
            'message':message,'measured_postcondition':relation}
    categories = [
        ('CAPABILITY_MISSING', ('requires a reorientation skill', 'contact skills that are not yet available'), ['retain_unfulfilled_condition', 'select_available_skill_or_report_gap']),
        ('REFERENCE_STALE', ('视觉引用', 'reference'), ['observe', 'select_current_reference']),
        ('TARGET_AMBIGUOUS', ('不唯一', 'ambiguous', '不同视角', '关联到地面'), ['select_visual_reference']),
        ('TARGET_NOT_FOUND', ('未找到', 'No observed target depth'), ['change_view_or_detector']),
        ('NO_FREE_SPACE', ('空位', '格位', '料箱格网', 'fitting the requested support'), ['inspect_destination', 'change_region_or_orientation', 'rearrange_obstacles_if_goal_allows']),
        ('NO_IK', ('Arena IK did not reach',), ['try_other_pose']),
        ('EMPTY_GRASP', ('did not lift', 'lift verification'), ['reobserve_then_regrasp']),
        ('NOT_HOLDING', ('没有确认', '夹持状态已改变'), ['reobserve_before_regrasp']),
        ('NO_CANDIDATE', ('no grasp candidates', 'no placement candidates'), ['change_view_or_model']),
        ('MODEL_UNAVAILABLE', ('timed out', 'Timeout', 'Connection', '服务调用失败'), ['retry_provider_or_change_model']),
    ]
    for code, fragments, recovery in categories:
        if any(part.lower() in message.lower() for part in fragments): break
    else: code, recovery = 'EXECUTION_FAILED', ['inspect_result']
    if holding.get('verified'):
        recovery = ['preserve_grasp', 'place_held'] + recovery
    return {'code': code, 'phase': phase, 'holding_verified': bool(holding.get('verified')),
            'suggested_recovery': recovery, 'message': message}
