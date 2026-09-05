/* eslint-disable @typescript-eslint/require-await -- Keep the async repository API and rejected-promise errors while memory operations execute atomically without yielding. */
import { Injectable } from '@nestjs/common';

export type IdempotencyKind = 'ingest' | 'publish' | 'physical';
@Injectable()
export class IdempotencyRepository {
  private readonly records = new Map<
    string,
    { eventId: string; kind: IdempotencyKind; createdAt: string }
  >();
  async findEventIdByKey(key: string): Promise<string | null> {
    return this.records.get(key)?.eventId ?? null;
  }
  has(key: string): boolean {
    return this.records.has(key);
  }
  /** Synchronous claim so event insertion can commit without yielding. */
  claim(key: string, eventId: string, kind: IdempotencyKind, nowIso: string): boolean {
    if (this.records.has(key)) return false;
    this.records.set(key, { eventId, kind, createdAt: nowIso });
    return true;
  }
  async insertIfAbsent(
    key: string,
    eventId: string,
    kind: IdempotencyKind,
    nowIso: string,
  ): Promise<boolean> {
    return this.claim(key, eventId, kind, nowIso);
  }
}
