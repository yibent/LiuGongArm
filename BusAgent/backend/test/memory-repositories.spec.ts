import { describe, expect, it } from 'vitest';
import { EventsRepository } from '../src/persistence/repositories/events.repository.js';
import { IdempotencyRepository } from '../src/persistence/repositories/idempotency.repository.js';
import { TasksRepository } from '../src/persistence/repositories/tasks.repository.js';
import { DeliveriesRepository } from '../src/persistence/repositories/deliveries.repository.js';
import { DeadLettersRepository } from '../src/persistence/repositories/dead-letters.repository.js';
import { SnapshotsRepository } from '../src/persistence/repositories/snapshots.repository.js';
import { RegistrationsRepository } from '../src/persistence/repositories/registrations.repository.js';
import { SystemClock } from '../src/common/clock.js';
import { TaskService } from '../src/bus/tasks/task.service.js';
import type { BusEvent } from '../src/protocol/envelope.js';

const now = '2026-09-05T00:00:00.000Z';
const opts = {
  receivedAt: now,
  trace: { traceId: 'trace', hostId: 'host', receivedAt: now },
  status: 'ingested' as const,
};
const claim = { key: 'shared', kind: 'publish' as const, nowIso: now };
function event(eventId: string): BusEvent {
  return {
    eventId,
    eventType: 'test.event',
    appId: 'app',
    sourceAgentId: 'agent',
    correlationId: 'conversation',
    priority: 1,
    createdAt: now,
    payload: { text: 'original' },
  };
}

describe('process-scoped repositories', () => {
  it('atomically stores one event for concurrent publishers sharing a key', async () => {
    const keys = new IdempotencyRepository();
    const events = new EventsRepository(keys);
    const results = await Promise.all(
      Array.from({ length: 50 }, (_, i) =>
        events.insert(event(`event-${i}`), opts, claim),
      ),
    );
    expect(results.filter((r) => r === 'claimed')).toHaveLength(1);
    expect(await keys.findEventIdByKey(claim.key)).toBe('event-0');
    expect(await events.findById('event-1')).toBeNull();
    expect(await keys.insertIfAbsent(claim.key, 'other', 'physical', now)).toBe(false);
    expect(await events.insert(event('event-0'), opts, claim)).toBe('replay');
  });

  it('does not reserve a key when event insertion fails', async () => {
    const keys = new IdempotencyRepository();
    const events = new EventsRepository(keys);
    await expect(
      events.insert({ ...event('bad'), payload: () => undefined }, opts, claim),
    ).rejects.toThrow();
    expect(await keys.findEventIdByKey(claim.key)).toBeNull();
    await events.insert(event('existing'), opts);
    await expect(events.insert(event('existing'), opts, claim)).rejects.toThrow(
      'Duplicate event ID',
    );
    expect(await keys.findEventIdByKey(claim.key)).toBeNull();
  });

  it('isolates stored event payloads from callers and starts new instances empty', async () => {
    const keys = new IdempotencyRepository();
    const events = new EventsRepository(keys);
    const input = event('event');
    await events.insert(input, opts, claim);
    (input.payload as { text: string }).text = 'changed';
    const output = await events.findById('event');
    expect(output?.payload).toEqual({ text: 'original' });
    (output!.payload as { text: string }).text = 'also changed';
    expect((await events.findById('event'))?.payload).toEqual({ text: 'original' });
    const freshKeys = new IdempotencyRepository();
    expect(await new EventsRepository(freshKeys).findById('event')).toBeNull();
    expect(await freshKeys.findEventIdByKey(claim.key)).toBeNull();
  });

  it('retains task version and cancellation gates until restart', async () => {
    const repo = new TasksRepository();
    const tasks = new TaskService(repo, new SystemClock());
    await tasks.ensureTask('task', 'app', 1);
    await tasks.ensureTask('task', 'app', 2);
    expect(await tasks.checkVersion('task', 1)).toBe('stale');
    await tasks.cancelTask('task');
    expect(await tasks.checkVersion('task', 2)).toBe('cancelled');
    const read = await repo.get('task');
    read!.state = 'active';
    expect(await tasks.checkVersion('task', 2)).toBe('cancelled');
    expect(await new TasksRepository().get('task')).toBeNull();
  });

  it('retains unknown physical-action state and dead letters only for this process', async () => {
    const deliveries = new DeliveriesRepository();
    await deliveries.create({
      id: 'delivery',
      eventId: 'event',
      agentId: 'arm',
      appId: 'app',
      queueName: 'queue',
      maxAttempts: 1,
      createdAt: now,
    });
    await deliveries.recordAttempt('delivery', {
      status: 'unknown',
      attempts: 1,
      updatedAt: now,
    });
    expect(await deliveries.hasUnknown('event', 'arm')).toBe(true);
    expect(await deliveries.hasUnknown('event', 'another-arm')).toBe(false);
    expect(await new DeliveriesRepository().hasUnknown('event', 'arm')).toBe(false);
    const letters = new DeadLettersRepository(new SystemClock());
    await letters.insert({
      eventId: 'event',
      appId: 'app',
      agentId: 'arm',
      reason: 'unknown',
      detail: null,
    });
    const rows = await letters.list();
    expect(rows).toHaveLength(1);
    await letters.markRedelivered(rows[0]!.id);
    expect(await new DeadLettersRepository(new SystemClock()).list()).toEqual([]);
  });

  it('keeps configuration snapshots and external leases isolated and ephemeral', async () => {
    const snapshots = new SnapshotsRepository();
    const payload = { setting: 1 };
    await snapshots.save('app', 'app', '1', 'hash', payload);
    payload.setting = 2;
    expect(await snapshots.findLatestByType('app', 'app')).toEqual({ setting: 1 });
    await snapshots.save('app', 'app', '2', 'hash2', payload);
    expect(await snapshots.findLatestByType('app', 'app')).toEqual({ setting: 2 });
    expect(await new SnapshotsRepository().findLatestByType('app', 'app')).toBeNull();
    const registrations = new RegistrationsRepository();
    await registrations.upsert({
      agentId: 'arm',
      endpointUrl: 'http://arm',
      instanceVersion: '1',
      leaseExpiresAtMs: 30000,
      registeredAtMs: 0,
      lastRenewedAtMs: 0,
    });
    expect(await registrations.findAll()).toHaveLength(1);
    expect(await new RegistrationsRepository().findAll()).toEqual([]);
    await registrations.delete('arm');
    expect(await registrations.findAll()).toEqual([]);
  });
});
