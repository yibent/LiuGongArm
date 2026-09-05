import { afterEach, describe, expect, it, vi } from 'vitest';
import { semanticFrame } from '../src/apps/desktop-robot/semantic-understanding.js';
import { buildPlan } from '../src/apps/desktop-robot/planner-agent.js';
import { validatePlan } from '../src/apps/desktop-robot/plan-validator-node.js';
import { RobotAdapterNode, HttpRobotAdapter } from '../src/apps/desktop-robot/executor-agent.js';
import { TaskConversation } from '../src/modules/dialogue/task-conversation.js';
import { makeEvent } from './helpers.js';
import type { InProcessEventContext } from '../src/adapters/in-process/agent-classes.js';

function context(plan: unknown, events: any[], id = 'integrated'): InProcessEventContext {
  return { event: makeEvent({ eventId: id, eventType: 'robot.execute.requested', taskId: `task_${id}`,
    payload: { plan } }), agentConfig: { appId: 'app', agentId: 'test', adapter: 'in-process',
      registrationKey: 'test', config: {} }, publish: async (input) => { events.push(input); } };
}
const reply = (body: unknown) => new Response(JSON.stringify(body));
describe('integrated skill sequences and memory', () => {
  afterEach(() => vi.unstubAllGlobals());
  it('supports an explicit sequence without inventing world coordinates', () => {
    const instruction = semanticFrame({ intent: 'sequence', actions: [
      { intent: 'target_move', category: 'block', color: 'red', offset_m: [0, 0, .1] },
      { intent: 'gripper', opening: 1 },
    ] }, '移到红方块上方十厘米，同时打开夹爪');
    const plan = buildPlan(instruction, 'test')!;
    expect(validatePlan(plan)).toEqual([]);
    expect(plan.steps.map((s) => s.skill)).toEqual(['move_relative_to_object', 'gripper']);
    expect(plan.steps[0]!.params.xyz_m).toBeUndefined();
  });
  it('clarifies the entire sequence if one action is unsupported', () => {
    const instruction = semanticFrame({ intent: 'sequence', actions: [
      { intent: 'home' }, { intent: 'pick_place', category: 'block' },
    ] }, '复位然后抓起并放置');
    expect(buildPlan(instruction, 'test')).toBeNull();
  });
  it.each([
    { intent: 'remember', category: 'block', color: 'red', memory_label: '零件A' },
    { intent: 'recall', memory_id: 'actual-id' },
    { intent: 'forget', memory_id: 'actual-id' },
    { intent: 'list_memories' }, { intent: 'scene_inventory' },
  ])('builds supported memory/observation action $intent', (frame) => {
    const plan = buildPlan(semanticFrame(frame, '测试'), 'test')!;
    expect(validatePlan(plan)).toEqual([]);
  });
  it('never sends movement after stop while grounding is pending', async () => {
    let enabled = true;
    const fetcher = vi.fn(async () => {
      enabled = false;
      return reply({ ok: true, target_world_m: [.2, 0, 1.2], observed_at: 5, cancel_epoch: 1 });
    });
    vi.stubGlobal('fetch', fetcher);
    const adapter = new HttpRobotAdapter('http://localhost:7861', 1000, () => enabled);
    const result = await adapter.execute('id', { id: 1, skill: 'move_relative_to_object',
      params: { target: { category: 'block' }, offset_m: [0, 0, .1] } }, 't', 1);
    expect(result.state).toBe('cancelled');
    expect(fetcher).toHaveBeenCalledTimes(1);
  });
  it.each(['completed', 'failed'])('waits for measured %s before dispatching next step', async (state) => {
    const events: any[] = [], commands: any[] = [];
    let finish!: () => void;
    const terminal = new Promise<void>((resolve) => { finish = resolve; });
    vi.stubGlobal('fetch', vi.fn(async (_url: string, options?: RequestInit) => {
      if (options?.method === 'POST') {
        const command = JSON.parse(String(options.body)); commands.push(command);
        return reply({ ok: true, state: 'accepted', command_id: command.command_id });
      }
      await terminal;
      return reply({ ok: state === 'completed', state, message: '实测结果' });
    }));
    const plan = buildPlan(semanticFrame({ intent: 'sequence', actions: [
      { intent: 'home' }, { intent: 'gripper', opening: 1 },
    ] }, '复位然后张开夹爪'), 'id')!;
    await new RobotAdapterNode().handle(context(plan, events));
    expect(commands).toHaveLength(1);
    finish();
    await vi.waitFor(() => expect(events.some((e) => e.event_type === `execution.${state}`)).toBe(true));
    expect(commands).toHaveLength(state === 'completed' ? 2 : 1);
    if (state === 'completed') {
      expect(events.filter((e) => e.event_type === 'execution.completed')).toHaveLength(1);
      expect(events.some((e) => e.payload?.step_completed)).toBe(true);
    }
  });
  it('tracks changing command ids within one task and rejects old-step events', () => {
    const memory = new TaskConversation(), ctx = context({}, []);
    const event = (type: string, step: number) => ({ ...ctx,
      event: makeEvent({ taskId: 't', eventType: type, payload: { step_id: step, command_id: `c${step}` } }) });
    expect(memory.observe(event('execution.accepted', 1))?.commandId).toBe('c1');
    expect(memory.observe(event('execution.accepted', 2))?.commandId).toBe('c2');
    expect(memory.observe(event('execution.completed', 1))).toBeUndefined();
    expect(memory.observe(event('execution.completed', 2))?.stage).toBe('completed');
  });
});
