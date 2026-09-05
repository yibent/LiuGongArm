import { afterEach, describe, expect, it, vi } from 'vitest';
import { parseInstruction } from '../src/apps/desktop-robot/instruction-agent.js';
import { buildPlan } from '../src/apps/desktop-robot/planner-agent.js';
import { validatePlan } from '../src/apps/desktop-robot/plan-validator-node.js';

afterEach(() => vi.unstubAllEnvs());
describe('opt-in industrial demonstration', () => {
  it('keeps placement unavailable by default', () => {
    vi.stubEnv('BUSAGENT_INDUSTRIAL_DEMO', '0');
    const instruction = parseInstruction('把金属柱放入篮子');
    expect(instruction.needs_clarification).toBe(true);
    expect(buildPlan(instruction, 'demo')).toBeNull();
  });
  it('routes placement through one scripted grasp transaction', () => {
    vi.stubEnv('BUSAGENT_INDUSTRIAL_DEMO', '1');
    const instruction = parseInstruction('把金属柱放入篮子');
    expect(instruction.needs_clarification).toBe(false);
    const plan = buildPlan(instruction, 'demo')!;
    expect(plan.steps).toHaveLength(1);
    expect(plan.steps[0]).toMatchObject({skill:'grasp', params:{demo_stack:true}});
    expect(validatePlan(plan)).toEqual([]);
  });
});
