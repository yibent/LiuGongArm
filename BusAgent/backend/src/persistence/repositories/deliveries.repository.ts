/* eslint-disable @typescript-eslint/require-await -- Keep the async repository API and rejected-promise errors while memory operations execute atomically without yielding. */
import { Injectable } from '@nestjs/common';

export type DeliveryStatus =
  | 'queued'
  | 'delivered'
  | 'retrying'
  | 'dead'
  | 'cancelled'
  | 'skipped'
  | 'unknown';
interface DeliveryRecord {
  id: string;
  eventId: string;
  appId: string;
  agentId: string;
  queueName: string;
  maxAttempts: number;
  createdAt: string;
  status: DeliveryStatus;
  attempts: number;
  lastError?: string;
  nextAttemptAt?: string;
  deliveredAt?: string;
  updatedAt: string;
}
@Injectable()
export class DeliveriesRepository {
  private readonly records = new Map<string, DeliveryRecord>();
  async create(record: {
    id: string;
    eventId: string;
    appId: string;
    agentId: string;
    queueName: string;
    maxAttempts: number;
    createdAt: string;
  }): Promise<void> {
    this.records.set(record.id, {
      ...record,
      status: 'queued',
      attempts: 0,
      updatedAt: record.createdAt,
    });
  }
  async recordAttempt(
    id: string,
    patch: {
      status: DeliveryStatus;
      attempts: number;
      lastError?: string;
      nextAttemptAt?: string;
      deliveredAt?: string;
      updatedAt: string;
    },
  ): Promise<void> {
    const record = this.records.get(id);
    if (record) Object.assign(record, patch);
  }
  async hasUnknown(eventId: string, agentId: string): Promise<boolean> {
    return [...this.records.values()].some(
      (r) => r.eventId === eventId && r.agentId === agentId && r.status === 'unknown',
    );
  }
}
