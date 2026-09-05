import { afterEach, describe, expect, it, vi } from 'vitest';
import { isStandaloneHome } from '../src/apps/desktop-robot/direct-command.js';
import { InstructionUnderstandingNode } from '../src/apps/desktop-robot/instruction-agent.js';
import { HostConfig } from '../src/config/host-config.js';
import {
  guardedAcknowledgement,
  hasPrematureActionClaim,
  isReadOnlyInteraction,
} from '../src/modules/dialogue/acknowledgement.js';
import { makeEvent } from './helpers.js';
import type { InProcessEventContext } from '../src/adapters/in-process/agent-classes.js';

describe('request factual boundaries', () => {
  afterEach(() => vi.unstubAllGlobals());
  it.each(['复位。', '先复位一下。', '把机械臂复位', '好，现在请你把机械臂归位。'])(
    'recognizes standalone home: %s',
    (text) => {
      expect(isStandaloneHome(text)).toBe(true);
    },
  );
  it.each([
    '不要复位',
    '复位了吗',
    '复位后抓取红方块',
    '同意回准备姿态',
    '先别复位',
    '复位，算了，停止',
    '复位到昨天的位置',
    '归位同时打开夹爪',
  ])('leaves ambiguous or compound requests to semantics: %s', (text) => {
    expect(isStandaloneHome(text)).toBe(false);
  });
  it('dispatches explicit home even when semantic provider is unavailable', async () => {
    const fetcher = vi.fn(() => {
      throw new Error('provider unavailable');
    });
    vi.stubGlobal('fetch', fetcher);
    const publish = vi.fn(async (_input: unknown) => {});
    const ctx = {
      event: makeEvent({ eventType: 'intent.created', payload: { text: '复位。' } }),
      agentConfig: { config: {}, agentId: 'nlu' },
      publish,
    } as unknown as InProcessEventContext;
    await new InstructionUnderstandingNode(
      HostConfig.fromEnv({ DASHSCOPE_API_KEY: 'test' }),
    ).handle(ctx);
    await vi.waitFor(() => expect(publish).toHaveBeenCalled());
    expect(fetcher).not.toHaveBeenCalled();
    expect(publish.mock.calls[0]?.[0]).toMatchObject({
      event_type: 'instruction.parsed',
      payload: {
        intent: 'motion',
        motion: { skill: 'home' },
        needs_clarification: false,
      },
    });
  });
  it.each(['已复位完成。', '正在执行复位。', '已到初始姿态并停稳。', '复位完成。'])(
    'rejects premature acknowledgement %s',
    (text) => {
      expect(hasPrematureActionClaim(text)).toBe(true);
    },
  );
  it('does not leak split false claims to UI or TTS and repairs using LLM callback', async () => {
    const repair = vi.fn(async () => '我先核对这条复位指令。');
    async function* stream() {
      yield '已复';
      yield '位完';
      yield '成。';
    }
    const received = [];
    for await (const text of guardedAcknowledgement(stream(), repair))
      received.push(text);
    expect(received).toEqual(['我先核对这条复位指令。']);
    expect(repair).toHaveBeenCalledOnce();
  });
  it('suppresses a second unsafe model reply rather than inventing a fixed success', async () => {
    async function* stream() {
      yield '正在执行复位。';
    }
    const received = [];
    for await (const text of guardedAcknowledgement(
      stream(),
      async () => '已复位完成。',
    ))
      received.push(text);
    expect(received).toEqual([]);
  });
  it('permits historical status only for standalone read-only questions', () => {
    expect(isReadOnlyInteraction('复位了吗？')).toBe(true);
    expect(isReadOnlyInteraction('先复位一下。')).toBe(false);
    expect(isReadOnlyInteraction('复位了吗，然后抓取红方块')).toBe(false);
  });
});
