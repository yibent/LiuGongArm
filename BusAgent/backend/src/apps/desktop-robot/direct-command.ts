/** Exact, standalone commands only. Never extract an action from a longer sentence. */
export function isStandaloneHome(text: string): boolean {
  const value = text.replace(/[\s，。！？,.!?]/g, '');
  return /^(?:嗯|呃|好|好的)?(?:现在)?(?:请|请你|帮我)?(?:先)?(?:把)?(?:机械臂|手臂)?(?:进行)?(?:复位|归位)(?:一下)?(?:吧)?$/.test(
    value,
  );
}
