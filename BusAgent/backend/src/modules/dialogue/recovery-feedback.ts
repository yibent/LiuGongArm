function record(value: unknown): Record<string, unknown> {
  return value && typeof value === 'object' ? value as Record<string, unknown> : {};
}
export function recoveryPermissions(payload: unknown) {
  const result = record(record(record(payload).result).result);
  const proposal = record(result.preparation);
  return {
    preparation: typeof proposal.id === 'string' && proposal.requires_confirmation === true,
    retry: result.retry_available === true,
  };
}
export function failureFeedbackBoundary(payload: unknown): string {
  const allowed = recoveryPermissions(payload);
  return '本次动作未完成，只解释本次事件的具体原因。' +
    (allowed.preparation
      ? '本次存在已检查的准备路径：说明可先回初始准备姿态并保持当前夹爪开度，询问是否同意；不保证之后能抓取。'
      : '本次没有可执行的准备路径。只报告失败原因，不添加复位、回初始位、开爪等建议，不询问同意。') +
    (allowed.retry ? '本次允许重新观察后重试，可询问是否重试。' : '本次不支持重试，不建议重试。');
}
export async function* guardedFailureFeedback(source: AsyncIterable<string>, payload: unknown) {
  const allowed = recoveryPermissions(payload);
  let pending = '';
  const permitted = (sentence: string) => {
    const advice = /建议|是否|同意|先回|请先|可以先|要不要|不妨|下次|重新尝试|再试/.test(sentence);
    if (!advice) return true;
    if (!allowed.preparation && /初始|准备位|准备姿态|复位|归位|回位|夹爪|开爪/.test(sentence)) return false;
    if (!allowed.retry && /重试|再试|重新尝试/.test(sentence)) return false;
    if (!allowed.preparation && !allowed.retry) return false;
    return true;
  };
  for await (const delta of source) {
    pending += delta;
    let match: RegExpExecArray | null;
    while ((match = /[。！？!?\n]/.exec(pending))) {
      const sentence = pending.slice(0, match.index + 1);
      pending = pending.slice(match.index + 1);
      if (permitted(sentence)) yield sentence;
    }
  }
  if (pending.trim() && permitted(pending)) yield pending;
}
