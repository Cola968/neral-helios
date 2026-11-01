import assert from 'node:assert/strict';
import test from 'node:test';
import worker from './worker.mjs';

function fixture() {
  const values = new Map();
  const writes = [];
  const env = { HEARTBEAT: {
    async get(key) { return values.get(key) ?? null; },
    async put(key, val) { writes.push([key, val]); values.set(key, val); },
  }};
  return { env, writes, values };
}

test('scheduled tick saves bounded timestamp only, no source or credentials', async () => {
  const { env, writes } = fixture();
  await worker.scheduled({ cron: '*/15 * * * *', scheduledTime: Date.parse('2026-11-01T15:00:00.000Z') }, env, {});
  assert.deepEqual(writes, [['last_tick', '2026-11-01T15:00:00.000Z']]);
});

test('health check is read-only, truthful and does not claim coding or training', async () => {
  const { env, writes, values } = fixture();
  values.set('last_tick', '2026-11-01T15:00:00.000Z');
  const response = await worker.fetch(new Request('https://worker.example/health'), env, {});
  assert.equal(response.status, 200);
  assert.equal(response.headers.get('cache-control'), 'no-store');
  const body = await response.json();
  assert.equal(body.last_tick, '2026-11-01T15:00:00.000Z');
  assert.equal(body.ai_code_agent_running, false);
  assert.equal(body.private_github_connected, false);
  assert.equal(body.gpu_training, false);
  assert.deepEqual(writes, []);
});

test('worker cannot be remotely triggered or made to run code', async () => {
  const { env, writes } = fixture();
  for (const url of ['https://worker.example/run','https://worker.example/execute','https://worker.example/jobs','https://worker.example/?cmd=rm']) {
    const r = await worker.fetch(new Request(url), env, {});
    assert.equal(r.status, 404);
  }
  const post = await worker.fetch(new Request('https://worker.example/health', {method:'POST'}), env, {});
  assert.equal(post.status, 405);
  assert.deepEqual(writes, []);
});

test('missing KV does not masquerade as an active scheduler', async () => {
  const r = await worker.fetch(new Request('https://worker.example/health'), {}, {});
  assert.equal(r.status, 503);
  assert.equal((await r.json()).scheduler, 'unavailable');
});

test('invalid scheduled timestamp does not write a false heartbeat', async () => {
  const { env, writes } = fixture();
  await assert.rejects(worker.scheduled({ cron:'*/15 * * * *', scheduledTime:Number.NaN }, env, {}));
  assert.deepEqual(writes, []);
});
