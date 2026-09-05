import { describe, expect, it } from 'vitest';
import { failureFeedbackBoundary, guardedFailureFeedback } from '../src/modules/dialogue/recovery-feedback.js';
async function collect(parts: string[], payload: unknown) {
  async function* stream() { for (const part of parts) yield part; }
  let result = '';
  for await (const part of guardedFailureFeedback(stream(), payload)) result += part;
  return result;
}
describe('factual recovery feedback', () => {
  it('removes an unsupported suggestion before UI and TTS even across chunks', async () => {
    expect(await collect(['抓取失败，姿态不可达。建议先回', '初始位，保持夹爪打开，是否同意？'], {}))
      .toBe('抓取失败，姿态不可达。');
    expect(failureFeedbackBoundary({})).not.toContain('存在已检查的准备路径');
  });
  it('keeps a real preparation question with its controller proposal', async () => {
    const text = '建议先回初始准备姿态并保持当前夹爪开度，是否同意？';
    expect(await collect([text], { result: { result: { preparation: { id: 'p1', requires_confirmation: true } } } }))
      .toBe(text);
  });
});
