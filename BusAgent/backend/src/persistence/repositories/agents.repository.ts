/* eslint-disable @typescript-eslint/require-await -- Keep the async repository API and rejected-promise errors while memory operations execute atomically without yielding. */
import { Injectable } from '@nestjs/common';
import type { InstalledAgent } from '../../registry/installed-agent.js';
import { Clock } from '../../common/clock.js';
@Injectable()
export class AgentsRepository {
  private records: Array<InstalledAgent & { installedAt: string }> = [];
  constructor(private readonly clock: Clock) {}
  async replaceAll(installed: InstalledAgent[]): Promise<void> {
    const installedAt = this.clock.now();
    this.records = structuredClone(installed.map((a) => ({ ...a, installedAt })));
  }
  async findAll(): Promise<Array<InstalledAgent & { installedAt: string }>> {
    return structuredClone(this.records);
  }
}
