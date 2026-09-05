/* eslint-disable @typescript-eslint/require-await -- Keep the async repository API and rejected-promise errors while memory operations execute atomically without yielding. */
import { Injectable } from '@nestjs/common';
import { randomUUID } from 'node:crypto';
import { Clock } from '../../common/clock.js';
@Injectable()
export class AuditRepository {
  private readonly records: Array<{
    id: string;
    action: string;
    entityType: string;
    entityId: string;
    detail: Record<string, unknown> | null;
    createdAt: string;
  }> = [];
  constructor(private readonly clock: Clock) {}
  async append(
    action: string,
    entityType: string,
    entityId: string,
    detail: Record<string, unknown> | null,
  ): Promise<void> {
    this.records.push({
      id: randomUUID(),
      action,
      entityType,
      entityId,
      detail: structuredClone(detail),
      createdAt: this.clock.now(),
    });
  }
}
