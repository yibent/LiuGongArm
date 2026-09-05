/* eslint-disable @typescript-eslint/require-await -- Keep the async repository API and rejected-promise errors while memory operations execute atomically without yielding. */
import { Injectable } from '@nestjs/common';
import type { BusEvent } from '../../protocol/envelope.js';
import type { TraceInfo } from '../../protocol/trace.js';
import {
  IdempotencyRepository,
  type IdempotencyKind,
} from './idempotency.repository.js';
export type EventStatus = 'ingested' | 'routed' | 'stale' | 'dropped';
export type IdempotencyClaimResult = 'claimed' | 'replay';
export interface EventInsertClaim {
  key: string;
  kind: IdempotencyKind;
  nowIso: string;
}
@Injectable()
export class EventsRepository {
  private readonly records = new Map<
    string,
    { event: BusEvent; receivedAt: string; trace: TraceInfo; status: EventStatus }
  >();
  constructor(private readonly idempotency: IdempotencyRepository) {}
  async insert(
    event: BusEvent,
    opts: { receivedAt: string; trace: TraceInfo; status: EventStatus },
    claim?: EventInsertClaim,
  ): Promise<IdempotencyClaimResult> {
    if (claim && this.idempotency.has(claim.key)) return 'replay';
    if (this.records.has(event.eventId))
      throw new Error(`Duplicate event ID: ${event.eventId}`);
    // Clone before claiming: invalid input cannot leave a key without an event.
    const record = structuredClone({ event, ...opts });
    if (
      claim &&
      !this.idempotency.claim(claim.key, event.eventId, claim.kind, claim.nowIso)
    )
      return 'replay';
    this.records.set(event.eventId, record);
    return 'claimed';
  }
  async findById(eventId: string): Promise<BusEvent | null> {
    return structuredClone(this.records.get(eventId)?.event ?? null);
  }
  async markStatus(eventId: string, status: EventStatus): Promise<void> {
    const record = this.records.get(eventId);
    if (record) record.status = status;
  }
}
