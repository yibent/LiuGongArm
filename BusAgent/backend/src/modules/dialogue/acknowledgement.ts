/** This classifies permission to report history, not permission to execute. */
export function isReadOnlyInteraction(text: string): boolean {
  const value = text.replace(/[\s，。！？,.!?]/g, '');
  if (/然后|接着|之后|顺便|再把|帮我|请把|同时/.test(value)) return false;
  return /^(?:嗯|呃|好|好的)?(?:现在|当前)?(?:你|机械臂|它)?(?:有什么能力|有什么功能|能做什么|支持什么|在做什么|现在在做什么|状态如何|好了没|好了没有|完成了吗|复位了吗|归位了吗|动了吗|为什么失败|刚才为什么失败)$/.test(
    value,
  );
}

// Check complete sentences before either UI or TTS sees them. This is a small
// factual boundary for unparsed requests, not an alternative semantic parser.
export function hasPrematureActionClaim(text: string): boolean {
  const value = text.replace(/尚未|还未|没有|不能|不会|不代表|未确认|未开始/g, '否定');
  return /(?:已|已经)[^。！？\n]{0,24}(?:完成|到位|停稳|复位|归位|开始|执行|张开|打开|闭合|抓住|抓起|移动|停止|取消|排队)|(?:正在|开始|开始了)[^。！？\n]{0,16}(?:执行|复位|归位|抓取|移动|旋转|张开|闭合|抬升)|^(?:复位|归位|动作|抓取|移动).{0,8}(?:完成|成功|到位)|^(?:完成了|完成|已完成|已到位)/.test(
    value,
  );
}

export async function* guardedAcknowledgement(
  source: AsyncIterable<string>,
  repair: (invalid: string) => Promise<string>,
): AsyncGenerator<string> {
  // Quick acknowledgements are one short sentence; ordinary chat and factual
  // execution events keep their existing streaming path.
  let sentence = '';
  let repaired = false;
  for await (const chunk of source) {
    sentence += chunk;
    if (!/[。！？.!?\n]$/.test(sentence)) continue;
    if (hasPrematureActionClaim(sentence)) {
      if (!repaired) {
        repaired = true;
        const corrected = await repair(sentence);
        if (corrected.trim() && !hasPrematureActionClaim(corrected)) yield corrected;
      }
    } else yield sentence;
    sentence = '';
  }
  if (sentence.trim()) {
    if (!hasPrematureActionClaim(sentence)) yield sentence;
    else if (!repaired) {
      const corrected = await repair(sentence);
      if (corrected.trim() && !hasPrematureActionClaim(corrected)) yield corrected;
    }
  }
}
