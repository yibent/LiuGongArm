/* eslint-disable @typescript-eslint/require-await -- Keep the async repository API and rejected-promise errors while memory operations execute atomically without yielding. */
import { Injectable } from '@nestjs/common';

export type TaskState = 'active' | 'cancelled' | 'completed';
export interface TaskRecord {
  taskId: string;
  appId: string;
  currentVersion: number;
  state: TaskState;
}
@Injectable()
export class TasksRepository {
  private readonly records = new Map<
    string,
    TaskRecord & { createdAt: string; updatedAt: string }
  >();
  async get(taskId: string): Promise<TaskRecord | null> {
    const record = this.records.get(taskId);
    if (!record) return null;
    const { appId, currentVersion, state } = record;
    return { taskId, appId, currentVersion, state };
  }
  async create(
    taskId: string,
    appId: string,
    version: number,
    nowIso: string,
  ): Promise<void> {
    if (!this.records.has(taskId))
      this.records.set(taskId, {
        taskId,
        appId,
        currentVersion: version,
        state: 'active',
        createdAt: nowIso,
        updatedAt: nowIso,
      });
  }
  async advanceVersion(taskId: string, version: number, nowIso: string): Promise<void> {
    const record = this.records.get(taskId);
    if (record) Object.assign(record, { currentVersion: version, updatedAt: nowIso });
  }
  async setState(taskId: string, state: TaskState, nowIso: string): Promise<void> {
    const record = this.records.get(taskId);
    if (record) Object.assign(record, { state, updatedAt: nowIso });
  }
}
