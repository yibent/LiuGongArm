/* eslint-disable @typescript-eslint/require-await -- Keep the async repository API and rejected-promise errors while memory operations execute atomically without yielding. */
import { Injectable } from '@nestjs/common';
import type { LeaseRecord } from '../../leases/lease-manager.service.js';
@Injectable()
export class RegistrationsRepository {
  private readonly records = new Map<string, LeaseRecord>();
  async upsert(record: LeaseRecord): Promise<void> {
    this.records.set(record.agentId, structuredClone(record));
  }
  async findAll(): Promise<LeaseRecord[]> {
    return structuredClone([...this.records.values()]);
  }
  async delete(agentId: string): Promise<void> {
    this.records.delete(agentId);
  }
}
