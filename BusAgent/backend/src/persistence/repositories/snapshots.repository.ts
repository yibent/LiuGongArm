/* eslint-disable @typescript-eslint/require-await -- Keep the async repository API and rejected-promise errors while memory operations execute atomically without yielding. */
import { Injectable } from '@nestjs/common';

@Injectable()
export class SnapshotsRepository {
  private readonly records = new Map<string, unknown>();
  async save(
    type: 'package' | 'app',
    entityKey: string,
    version: string,
    sha256: string,
    payload: unknown,
  ): Promise<void> {
    this.records.set(
      JSON.stringify([type, entityKey]),
      structuredClone({ version, sha256, payload }),
    );
  }
  async findLatestByType(type: 'package' | 'app', entityKey: string): Promise<unknown> {
    const record = this.records.get(JSON.stringify([type, entityKey])) as
      | { payload: unknown }
      | undefined;
    return structuredClone(record?.payload ?? null);
  }
}
