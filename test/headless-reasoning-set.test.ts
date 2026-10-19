/**
 * Integration test for the operator reasoning verbs against a REAL headless
 * child (full framework, no inference): `reasoning-get` / `reasoning-set`
 * over the IPC socket, the host-side settings ledger they write, the
 * operator_panel attribution of panel settings changes, persistence across a
 * restart (with the startup line that makes a restart's loaded state
 * visible), and that the agent's context is untouched by any of it.
 * Built for Aesop's intervention arm (2026-10).
 */
import { describe, test, expect, afterAll } from 'bun:test';
import { spawn, type ChildProcess } from 'node:child_process';
import { mkdtempSync, writeFileSync, rmSync, existsSync, readFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { connect as netConnect, type Socket } from 'node:net';
import { join, resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const TEST_DIR = dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = resolve(TEST_DIR, '..');
const INDEX_PATH = join(REPO_ROOT, 'src', 'index.ts');

const MINIMAL_RECIPE = {
  name: 'Reasoning Set Test',
  agent: {
    name: 'subject',
    systemPrompt: 'never asked to infer in this test',
  },
  modules: {
    subagents: false,
    lessons: false,
    retrieval: false,
    wake: false,
    workspace: false,
  },
};

function lineReader(socket: Socket): { events: Array<Record<string, unknown>>; stop: () => void } {
  const events: Array<Record<string, unknown>> = [];
  let buf = '';
  const handler = (chunk: Buffer): void => {
    buf += chunk.toString('utf-8');
    let i: number;
    while ((i = buf.indexOf('\n')) >= 0) {
      const line = buf.slice(0, i).trim();
      buf = buf.slice(i + 1);
      if (!line) continue;
      try { events.push(JSON.parse(line) as Record<string, unknown>); } catch { /* ignore malformed */ }
    }
  };
  socket.on('data', handler);
  return { events, stop: (): void => { socket.off('data', handler); } };
}

async function waitFor(check: () => boolean, timeoutMs: number, label: string): Promise<void> {
  const start = Date.now();
  while (Date.now() - start < timeoutMs) {
    if (check()) return;
    await new Promise((r) => setTimeout(r, 50));
  }
  throw new Error(`waitFor timed out after ${timeoutMs}ms: ${label}`);
}

async function connectSocket(path: string, timeoutMs = 3_000): Promise<Socket> {
  return new Promise((resolveConn, rejectConn) => {
    const s = netConnect(path);
    const timer = setTimeout(() => {
      s.destroy();
      rejectConn(new Error(`socket connect timeout: ${path}`));
    }, timeoutMs);
    s.once('connect', () => { clearTimeout(timer); resolveConn(s); });
    s.once('error', (err) => { clearTimeout(timer); rejectConn(err); });
  });
}

interface Child {
  proc: ChildProcess;
  sock: Socket;
  reader: { events: Array<Record<string, unknown>>; stop: () => void };
}

const tmpDir = mkdtempSync(join(tmpdir(), 'fkm-reasoning-set-'));
const recipePath = join(tmpDir, 'recipe.json');
const socketPath = join(tmpDir, 'ipc.sock');
const ledgerPath = join(tmpDir, 'settings-changes.jsonl');
writeFileSync(recipePath, JSON.stringify(MINIMAL_RECIPE), 'utf-8');

async function startChild(): Promise<Child> {
  const proc = spawn('bun', [INDEX_PATH, recipePath, '--headless'], {
    cwd: tmpDir,
    env: { ...process.env, ANTHROPIC_API_KEY: 'sk-test-reasoning', DATA_DIR: tmpDir, SETTINGS_LEDGER_PATH: '' },
    stdio: ['ignore', 'ignore', 'ignore'],
  });
  await waitFor(() => existsSync(socketPath), 15_000, 'socket file appears');
  const sock = await connectSocket(socketPath);
  const reader = lineReader(sock);
  await waitFor(
    () => reader.events.some((e) => e.type === 'lifecycle' && (e as { phase?: string }).phase === 'ready'),
    5_000,
    'lifecycle:ready',
  );
  return { proc, sock, reader };
}

async function stopChild(c: Child): Promise<void> {
  c.sock.write(JSON.stringify({ type: 'shutdown' }) + '\n');
  await waitFor(() => c.proc.exitCode !== null || !existsSync(socketPath), 10_000, 'child exits');
  await waitFor(() => c.proc.exitCode !== null, 10_000, 'child process exit');
  try { c.reader.stop(); c.sock.destroy(); } catch { /* noop */ }
}

let corrSeq = 0;
async function call(c: Child, msg: Record<string, unknown>, responseType: string): Promise<Record<string, unknown>> {
  const corrId = `rs-${++corrSeq}`;
  c.sock.write(JSON.stringify({ ...msg, corrId }) + '\n');
  await waitFor(() => c.reader.events.some((e) => e.type === responseType && e.corrId === corrId), 10_000, responseType);
  return c.reader.events.find((e) => e.type === responseType && e.corrId === corrId)!;
}

function ledger(): Array<Record<string, unknown>> {
  if (!existsSync(ledgerPath)) return [];
  return readFileSync(ledgerPath, 'utf-8').trim().split('\n').filter(Boolean).map((l) => JSON.parse(l));
}

let live: Child | null = null;
afterAll(async () => {
  if (live) {
    try { live.sock.write(JSON.stringify({ type: 'shutdown' }) + '\n'); } catch { /* noop */ }
    await new Promise((r) => setTimeout(r, 500));
    try { if (live.proc.exitCode === null) live.proc.kill('SIGKILL'); } catch { /* noop */ }
  }
  try { rmSync(tmpDir, { recursive: true, force: true }); } catch { /* noop */ }
});

describe('headless daemon — reasoning-get / reasoning-set', () => {
  test('flip over IPC, ledger every source, persist across restart, context untouched', async () => {
    live = await startChild();

    // Boot recorded what was loaded: reasoning keys (module) + core keys (framework).
    const boot = ledger();
    expect(boot.filter((l) => l.source === 'startup').map((l) => l.key)).toEqual(expect.arrayContaining([
      'reasoning_enabled', 'reasoning_display', 'contextBudgetTokens',
    ]));
    expect(boot.find((l) => l.key === 'reasoning_enabled')).toMatchObject({ source: 'startup', action: 'load', to: false });

    const got = await call(live, { type: 'reasoning-get' }, 'reasoning-get-result');
    expect(got).toMatchObject({ ok: true, reasoning: { reasoning_enabled: false, reasoning_display: 'summarized' }, idle: true });

    const coverageBefore = await call(live, { type: 'panel-request', op: 'context-coverage' }, 'panel-response');

    // The flip.
    const set = await call(live, { type: 'reasoning-set', enabled: true }, 'reasoning-set-result');
    expect(set).toMatchObject({ ok: true, before: { reasoning_enabled: false }, after: { reasoning_enabled: true } });
    const got2 = await call(live, { type: 'reasoning-get' }, 'reasoning-get-result');
    expect(got2).toMatchObject({ reasoning: { reasoning_enabled: true } });
    expect(ledger().filter((l) => l.source === 'operator_ipc')).toEqual([
      expect.objectContaining({ action: 'set', key: 'reasoning_enabled', from: false, to: true }),
    ]);

    // Validation errors come back ok:false and change nothing.
    const bad = await call(live, { type: 'reasoning-set', budgetTokens: 2048 }, 'reasoning-set-result');
    expect(bad.ok).toBe(false);

    // A panel settings change is attributed to the operator panel.
    const panel = await call(live, { type: 'panel-request', op: 'settings-update', params: { contextBudgetTokens: 150000 } }, 'panel-response');
    expect(panel.ok).toBe(true);
    expect(ledger().filter((l) => l.source === 'operator_panel')).toEqual([
      expect.objectContaining({ action: 'update', agent: 'subject', key: 'contextBudgetTokens', to: 150000 }),
    ]);

    // Nothing reached the agent's context.
    const coverageAfter = await call(live, { type: 'panel-request', op: 'context-coverage' }, 'panel-response');
    expect((coverageAfter.data as { totals: unknown }).totals).toEqual((coverageBefore.data as { totals: unknown }).totals);

    // Restart: the flipped state persists, and the boot line says so.
    await stopChild(live);
    live = null;
    const linesBefore = ledger().length;
    live = await startChild();
    const reboot = ledger().slice(linesBefore);
    expect(reboot.find((l) => l.key === 'reasoning_enabled')).toMatchObject({ source: 'startup', to: true });
    const got3 = await call(live, { type: 'reasoning-get' }, 'reasoning-get-result');
    expect(got3).toMatchObject({ reasoning: { reasoning_enabled: true } });

    // Revert, as the end of a block would.
    const revert = await call(live, { type: 'reasoning-set', enabled: false }, 'reasoning-set-result');
    expect(revert).toMatchObject({ ok: true, after: { reasoning_enabled: false } });

    // Every line has the declared shape.
    for (const l of ledger()) {
      expect(Object.keys(l).sort()).toEqual(
        (l.agent !== undefined ? ['action', 'agent', 'from', 'key', 'pid', 'source', 't', 'to'] : ['action', 'from', 'key', 'pid', 'source', 't', 'to']),
      );
    }
  }, 60_000);
});
