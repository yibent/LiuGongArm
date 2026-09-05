/* eslint-disable @typescript-eslint/require-await -- Keep the async repository API and rejected-promise errors while memory operations execute atomically without yielding. */
import { Injectable } from '@nestjs/common';
import { randomUUID } from 'node:crypto';
import { Clock } from '../../common/clock.js';
export interface DeadLetterEntry {
  eventId: string;
  appId: string;
  agentId: string;
  reason: string;
  detail: Record<string, unknown> | null;
}
@Injectable()
export class DeadLettersRepository {
  private readonly records = new Map<
    string,
    DeadLetterEntry & {
      id: string;
      status: 'dead' | 'redelivered';
      createdAt: string;
      redeliveredAt?: string;
    }
  >();
  constructor(private readonly clock: Clock) {}
  async insert(entry: DeadLetterEntry): Promise<void> {
    const id = randomUUID();
    this.records.set(id, {
      ...structuredClone(entry),
      id,
      status: 'dead',
      createdAt: this.clock.now(),
    });
  }
  async list(): Promise<
    Array<{ id: string; eventId: string; agentId: string; reason: string }>
  > {
    return [...this.records.values()].map(({ id, eventId, agentId, reason }) => ({
      id,
      eventId,
      agentId,
      reason,
    }));
  }
  async markRedelivered(id: string): Promise<void> {
    const record = this.records.get(id);
    if (record)
      Object.assign(record, { status: 'redelivered', redeliveredAt: this.clock.now() });
  }
}
