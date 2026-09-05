import { describe, expect, it } from 'vitest';
import { Test } from '@nestjs/testing';
import { HostRuntimeService } from '../src/app/host-runtime.service.js';
import { AppModule } from '../src/app.module.js';

/**
 * Verifies the whole Nest DI graph can be instantiated without booting the
 * host lifecycle: every provider is
 * resolvable by type and the forwardRef module cycle between Bus/Adapters
 * composes correctly.
 */
describe('DI graph', () => {
  it('resolves every provider in AppModule', async () => {
    const app = await Test.createTestingModule({ imports: [AppModule] }).compile();
    expect(app).toBeDefined();
    await app.close().catch(() => undefined);
  }, 15_000);

  it('boots the configured host to ready with empty memory storage', async () => {
    const app = await Test.createTestingModule({ imports: [AppModule] }).compile();
    try {
      await app.init();
      const runtime = app.get(HostRuntimeService);
      await runtime.awaitReady();
      expect(runtime.currentState).toBe('ready');
    } finally {
      await app.close();
    }
  }, 15_000);
});
