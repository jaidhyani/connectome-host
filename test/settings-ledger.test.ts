/**
 * The host-side settings-change ledger (src/settings-ledger.ts) and the
 * operator reasoning verbs (src/operator-reasoning.ts), built for Aesop's
 * intervention arm (2026-10): an operator flips reasoning_enabled on a sealed
 * schedule, and the reveal must show every OTHER change too.
 *
 * Covers: an operator flip lands on the next model call; mid-turn sets are
 * refused, not queued; every source is recorded (the agent's own tool, a
 * blanket reset, the operator panel, IPC, startup); and neither the ledger
 * nor the operator path ever touches the agent's context.
 */
import { describe, test, expect } from 'bun:test';
import { mkdtempSync, readFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import {
  SettingsLedger,
  instrumentFrameworkSettings,
  withSettingsSource,
  type SettingsChangeLine,
} from '../src/settings-ledger.js';
import { SettingsModule } from '../src/modules/settings-module.js';
import { operatorGetReasoning, operatorSetReasoning } from '../src/operator-reasoning.js';
import { LoggingAnthropicAdapter } from '../src/logging-adapter.js';

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

function memLedger(): { ledger: SettingsLedger; lines: SettingsChangeLine[] } {
  const lines: SettingsChangeLine[] = [];
  const ledger = new SettingsLedger('/nonexistent/ledger.jsonl', {
    now: () => new Date('2026-10-19T17:00:00.000Z'),
    write: (l) => lines.push(JSON.parse(l) as SettingsChangeLine),
  });
  return { ledger, lines };
}

/** A ModuleContext stand-in that records everything the module does with it.
 *  Only getState/setState exist: any attempt to reach the agent's context
 *  (pushEvent, addMessage, …) would be a TypeError, failing the test. */
function fakeCtx(saved?: unknown) {
  let state = saved;
  const calls: string[] = [];
  const ctx = {
    getState: <T>() => { calls.push('getState'); return state as T; },
    setState: (s: unknown) => { calls.push('setState'); state = JSON.parse(JSON.stringify(s)); },
  };
  return { ctx, calls, persisted: () => state };
}

async function startedModule(saved?: unknown) {
  const { ledger, lines } = memLedger();
  const mod = new SettingsModule({ ledger });
  const fc = fakeCtx(saved);
  await mod.start(fc.ctx as never);
  return { mod, lines, ...fc };
}

function fakeApp(mod: SettingsModule, statuses: Record<string, string>) {
  return {
    framework: {
      getAllModules: () => [{ name: 'tui' }, mod],
      getAllAgents: () => Object.entries(statuses).map(([name, status]) => ({ name, state: { status } })),
    },
  };
}

/** A fake Anthropic client that records each wire request and answers with
 *  a minimal end_turn message, so `complete()` and `stream()` both run
 *  membrane's real buildRequest + onRequest path. `stream()` is the agent's
 *  turn; `complete()` is compression and other background calls. */
function wireAdapter(logPath: string, getReasoning: () => { enabled: boolean; budgetTokens: number; display?: 'summarized' | 'omitted' }) {
  const adapter = new LoggingAnthropicAdapter(
    { apiKey: 'test', cacheKeepalive: { enabled: false } } as never,
    logPath,
    getReasoning,
  );
  const sent: Array<Record<string, unknown>> = [];
  (adapter as unknown as { client: unknown }).client = {
    messages: {
      create: async (params: Record<string, unknown>) => {
        sent.push(params);
        return {
          id: 'msg_test', type: 'message', role: 'assistant', model: params.model,
          content: [{ type: 'text', text: 'ok' }],
          stop_reason: 'end_turn', stop_sequence: null,
          usage: { input_tokens: 1, output_tokens: 1 },
        };
      },
      stream: async (params: Record<string, unknown>) => {
        sent.push(params);
        const events = [
          { type: 'message_start', message: { model: params.model, usage: { input_tokens: 1, output_tokens: 0 } } },
          { type: 'content_block_start', index: 0, content_block: { type: 'text', text: '' } },
          { type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text: 'ok' } },
          { type: 'content_block_stop', index: 0 },
          { type: 'message_delta', delta: { stop_reason: 'end_turn', stop_sequence: null }, usage: { output_tokens: 1 } },
          { type: 'message_stop' },
        ];
        return { async *[Symbol.asyncIterator]() { for (const e of events) yield e; } };
      },
    },
  };
  const agentTurn = () => adapter.stream(request, { onChunk: () => {} } as never);
  const background = () => adapter.complete(request);
  return { adapter, sent, agentTurn, background };
}

const request = {
  model: 'claude-fable-5',
  maxTokens: 64,
  messages: [{ participant: 'user', content: [{ type: 'text', text: 'hi' }] }],
} as never;

function lastLogLine(path: string): Record<string, unknown> {
  const lines = readFileSync(path, 'utf-8').trim().split('\n');
  return JSON.parse(lines[lines.length - 1]!) as Record<string, unknown>;
}

// ---------------------------------------------------------------------------
// operator flip → next call
// ---------------------------------------------------------------------------

describe('operator reasoning flip', () => {
  test('an operator flip lands on the next model call, with no restart', async () => {
    const dir = mkdtempSync(join(tmpdir(), 'arm-flip-'));
    try {
      const { mod } = await startedModule();
      const logPath = join(dir, 'llm-calls.jsonl');
      const { sent, agentTurn } = wireAdapter(logPath, () => mod.getReasoning());
      const app = fakeApp(mod, { aesop: 'idle' });

      await agentTurn();
      expect(sent[0]!.thinking).toBeUndefined();

      const res = operatorSetReasoning(app, { type: 'reasoning-set', corrId: 'c1', enabled: true });
      expect(res.ok).toBe(true);

      await agentTurn();
      expect(sent[1]!.thinking).toEqual({ type: 'adaptive', display: 'summarized' });

      expect(operatorSetReasoning(app, { display: 'omitted' }).ok).toBe(true);
      await agentTurn();
      expect(sent[2]!.thinking).toEqual({ type: 'adaptive', display: 'omitted' });

      expect(operatorSetReasoning(app, { enabled: false }).ok).toBe(true);
      await agentTurn();
      expect(sent[3]!.thinking).toBeUndefined();
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  });

  test('the flip persists to module state, as the agent tool does', async () => {
    const { mod, persisted } = await startedModule();
    operatorSetReasoning(fakeApp(mod, { aesop: 'idle' }), { enabled: true });
    expect((persisted() as { reasoning: { enabled: boolean } }).reasoning.enabled).toBe(true);
  });

  test('result carries before and after in wire names', async () => {
    const { mod } = await startedModule();
    const res = operatorSetReasoning(fakeApp(mod, { aesop: 'idle' }), { enabled: true });
    expect(res).toMatchObject({
      ok: true,
      before: { reasoning_enabled: false, reasoning_display: 'summarized' },
      after: { reasoning_enabled: true, reasoning_display: 'summarized' },
    });
  });

  test('reasoning-get reports the levers and idleness', async () => {
    const { mod } = await startedModule();
    expect(operatorGetReasoning(fakeApp(mod, { aesop: 'idle' }))).toEqual({
      ok: true,
      reasoning: { reasoning_enabled: false, reasoning_display: 'summarized' },
      idle: true,
      statuses: { aesop: 'idle' },
    });
    expect(operatorGetReasoning(fakeApp(mod, { aesop: 'streaming' }))).toMatchObject({ idle: false });
  });

  test('only the two levers are accepted; bad values apply nothing', async () => {
    const { mod, lines } = await startedModule();
    const app = fakeApp(mod, { aesop: 'idle' });
    const before = lines.length;
    expect(operatorSetReasoning(app, { budgetTokens: 4096 })).toMatchObject({ ok: false });
    expect(operatorSetReasoning(app, { enabled: 'yes' })).toMatchObject({ ok: false });
    expect(operatorSetReasoning(app, { display: 'verbose' })).toMatchObject({ ok: false });
    expect(operatorSetReasoning(app, {})).toMatchObject({ ok: false });
    expect(mod.getReasoning()).toMatchObject({ enabled: false, display: 'summarized', budgetTokens: 8192 });
    expect(lines.length).toBe(before);
  });

  test('settings module missing → ok:false, never a throw', () => {
    const app = { framework: { getAllModules: () => [], getAllAgents: () => [] } };
    expect(operatorSetReasoning(app, { enabled: true })).toEqual({ ok: false, error: 'settings module not loaded' });
    expect(operatorGetReasoning(app)).toEqual({ ok: false, error: 'settings module not loaded' });
  });
});

// ---------------------------------------------------------------------------
// mid-turn
// ---------------------------------------------------------------------------

describe('mid-turn handling: refused, not queued', () => {
  for (const status of ['inferring', 'streaming', 'waiting_for_tools', 'ready']) {
    test(`refuses while an agent is ${status}, and changes nothing`, async () => {
      const { mod, lines } = await startedModule();
      const before = lines.length;
      const res = operatorSetReasoning(fakeApp(mod, { aesop: status }), { enabled: true });
      expect(res).toEqual({ ok: false, error: 'busy', statuses: { aesop: status } });
      expect(mod.getReasoning().enabled).toBe(false);
      expect(lines.length).toBe(before);
    });
  }

  test('any busy agent blocks the set, not only the primary', async () => {
    const { mod } = await startedModule();
    const res = operatorSetReasoning(fakeApp(mod, { aesop: 'idle', summarizer: 'inferring' }), { enabled: true });
    expect(res).toMatchObject({ ok: false, error: 'busy' });
    expect(mod.getReasoning().enabled).toBe(false);
  });

  test('nothing is queued: a refused set never applies later', async () => {
    const { mod } = await startedModule();
    const statuses = { aesop: 'streaming' };
    const app = fakeApp(mod, statuses);
    operatorSetReasoning(app, { enabled: true });
    statuses.aesop = 'idle';
    await new Promise((r) => setTimeout(r, 20));
    expect(mod.getReasoning().enabled).toBe(false);
  });
});

// ---------------------------------------------------------------------------
// ledger: every source
// ---------------------------------------------------------------------------

describe('settings ledger records every source', () => {
  test('startup: the loaded reasoning state, one line per key', async () => {
    const { lines } = await startedModule({ reasoning: { enabled: true, display: 'omitted' } });
    expect(lines).toEqual([
      { t: '2026-10-19T17:00:00.000Z', pid: process.pid, source: 'startup', action: 'load', key: 'reasoning_enabled', from: null, to: true },
      { t: '2026-10-19T17:00:00.000Z', pid: process.pid, source: 'startup', action: 'load', key: 'reasoning_budget_tokens', from: null, to: 8192 },
      { t: '2026-10-19T17:00:00.000Z', pid: process.pid, source: 'startup', action: 'load', key: 'reasoning_display', from: null, to: 'omitted' },
    ]);
  });

  test('operator_ipc: the reasoning-set verb', async () => {
    const { mod, lines } = await startedModule();
    lines.length = 0;
    operatorSetReasoning(fakeApp(mod, { aesop: 'idle' }), { enabled: true });
    expect(lines).toEqual([
      expect.objectContaining({ source: 'operator_ipc', action: 'set', key: 'reasoning_enabled', from: false, to: true }),
    ]);
  });

  test("agent_tool: the agent's own agent_settings update", async () => {
    const { mod, lines } = await startedModule();
    lines.length = 0;
    mod.getAgentSettingsExtension().update('aesop', { reasoning_enabled: true, reasoning_display: 'omitted' });
    expect(lines).toEqual([
      expect.objectContaining({ source: 'agent_tool', action: 'update', agent: 'aesop', key: 'reasoning_enabled', from: false, to: true }),
      expect.objectContaining({ source: 'agent_tool', action: 'update', agent: 'aesop', key: 'reasoning_display', from: 'summarized', to: 'omitted' }),
    ]);
  });

  test('agent_tool: a blanket reset after an operator flip is recorded, key by key', async () => {
    const { mod, lines } = await startedModule();
    operatorSetReasoning(fakeApp(mod, { aesop: 'idle' }), { enabled: true });
    lines.length = 0;
    // `agent_settings reset` with no `settings` list → ext.reset(agent) with no keys.
    mod.getAgentSettingsExtension().reset('aesop');
    expect(mod.getReasoning().enabled).toBe(false); // semantics unchanged: it DOES undo the flip
    expect(lines.map((l) => [l.source, l.action, l.key, l.from, l.to])).toEqual([
      ['agent_tool', 'reset', 'reasoning_enabled', true, false],
      ['agent_tool', 'reset', 'reasoning_budget_tokens', 8192, 8192],
      ['agent_tool', 'reset', 'reasoning_display', 'summarized', 'summarized'],
    ]);
  });

  test('agent_tool: a keyed reset records only its keys', async () => {
    const { mod, lines } = await startedModule();
    lines.length = 0;
    mod.getAgentSettingsExtension().reset('aesop', ['reasoning_display']);
    expect(lines.map((l) => l.key)).toEqual(['reasoning_display']);
  });

  test('a rejected agent update records nothing and applies nothing', async () => {
    const { mod, lines } = await startedModule();
    lines.length = 0;
    expect(() => mod.getAgentSettingsExtension().update('aesop', { reasoning_enabled: true, reasoning_display: 'loud' }))
      .toThrow();
    expect(mod.getReasoning().enabled).toBe(false);
    expect(lines).toEqual([]);
  });

  describe('core runtime settings (framework instance instrumentation)', () => {
    /** Mimics the framework: its tool dispatch calls the mutators through
     *  `this`, which must resolve to the instrumented instance methods. */
    class FakeFramework {
      s: Record<string, unknown> = {
        contextBudgetTokens: 233616, tailTokens: 80000, transitionPaceTokens: 4000,
        sameRoundThinkTextPolicy: 'private', transition: 'stable',
      };
      getAgentRuntimeSettings(_a: string) { return { ...this.s }; }
      updateAgentRuntimeSettings(_a: string, p: Record<string, unknown>) { Object.assign(this.s, p); return this.s; }
      resetAgentRuntimeSettings(_a: string, keys?: string[]) {
        for (const k of keys ?? ['contextBudgetTokens', 'tailTokens']) this.s[k] = k === 'tailTokens' ? 60000 : 200000;
        return this.s;
      }
      cancelAgentRuntimeSettingsTransition(_a: string) { this.s.transition = 'stable'; this.s.contextBudgetTokens = 210000; return this.s; }
      /** The framework's agent_settings dispatch, reduced to the call shape. */
      dispatchAgentSettingsTool(input: { action: string; patch?: Record<string, unknown> }) {
        if (input.action === 'update') return this.updateAgentRuntimeSettings('aesop', input.patch!);
        if (input.action === 'reset') return this.resetAgentRuntimeSettings('aesop');
        return this.cancelAgentRuntimeSettingsTransition('aesop');
      }
    }

    test("the agent's tool (internal this.* call) is recorded as agent_tool", () => {
      const { ledger, lines } = memLedger();
      const fw = new FakeFramework();
      instrumentFrameworkSettings(fw, ledger);
      fw.dispatchAgentSettingsTool({ action: 'update', patch: { contextBudgetTokens: 250000 } });
      expect(lines).toEqual([
        expect.objectContaining({ source: 'agent_tool', action: 'update', agent: 'aesop', key: 'contextBudgetTokens', from: 233616, to: 250000 }),
      ]);
    });

    test('a blanket core reset records every core key', () => {
      const { ledger, lines } = memLedger();
      const fw = new FakeFramework();
      instrumentFrameworkSettings(fw, ledger);
      fw.dispatchAgentSettingsTool({ action: 'reset' });
      expect(lines.map((l) => [l.source, l.action, l.key])).toEqual([
        ['agent_tool', 'reset', 'contextBudgetTokens'],
        ['agent_tool', 'reset', 'tailTokens'],
        ['agent_tool', 'reset', 'transitionPaceTokens'],
        ['agent_tool', 'reset', 'sameRoundThinkTextPolicy'],
      ]);
    });

    test('operator panel calls are recorded as operator_panel, and the scope unwinds', () => {
      const { ledger, lines } = memLedger();
      const fw = new FakeFramework();
      instrumentFrameworkSettings(fw, ledger);
      withSettingsSource('operator_panel', () => fw.updateAgentRuntimeSettings('aesop', { tailTokens: 90000 }));
      fw.dispatchAgentSettingsTool({ action: 'update', patch: { tailTokens: 70000 } });
      expect(lines.map((l) => [l.source, l.key, l.from, l.to])).toEqual([
        ['operator_panel', 'tailTokens', 80000, 90000],
        ['agent_tool', 'tailTokens', 90000, 70000],
      ]);
    });

    test('the scope unwinds even when the framework throws', () => {
      const { ledger, lines } = memLedger();
      const fw = new FakeFramework();
      let failNext = true;
      const real = fw.updateAgentRuntimeSettings.bind(fw);
      fw.updateAgentRuntimeSettings = (a: string, p: Record<string, unknown>) => {
        if (failNext) { failNext = false; throw new Error('budget too small'); }
        return real(a, p);
      };
      instrumentFrameworkSettings(fw, ledger);
      expect(() => withSettingsSource('operator_panel', () => fw.updateAgentRuntimeSettings('aesop', { tailTokens: 1 }))).toThrow();
      expect(lines).toEqual([]); // a rejected change is not a change
      fw.dispatchAgentSettingsTool({ action: 'update', patch: { tailTokens: 70000 } });
      expect(lines).toEqual([expect.objectContaining({ source: 'agent_tool', key: 'tailTokens', to: 70000 })]);
    });

    test('cancel is never invisible', () => {
      const { ledger, lines } = memLedger();
      const fw = new FakeFramework();
      fw.s.transition = 'converging';
      instrumentFrameworkSettings(fw, ledger);
      fw.dispatchAgentSettingsTool({ action: 'cancel' });
      expect(lines.map((l) => [l.action, l.key, l.from, l.to])).toEqual([
        ['cancel', 'contextBudgetTokens', 233616, 210000],
        ['cancel', 'transition', 'converging', 'stable'],
      ]);
    });

    test('instrumenting twice does not double-record', () => {
      const { ledger, lines } = memLedger();
      const fw = new FakeFramework();
      instrumentFrameworkSettings(fw, ledger);
      instrumentFrameworkSettings(fw, ledger);
      fw.dispatchAgentSettingsTool({ action: 'update', patch: { tailTokens: 1000 } });
      expect(lines.length).toBe(1);
    });
  });

  test('the file sink appends JSONL with owner-only permissions', () => {
    const dir = mkdtempSync(join(tmpdir(), 'arm-ledger-'));
    try {
      const path = join(dir, 'settings-changes.jsonl');
      const ledger = new SettingsLedger(path);
      ledger.record('operator_ipc', 'set', ['reasoning_enabled'], { reasoning_enabled: false }, { reasoning_enabled: true });
      ledger.record('agent_tool', 'reset', ['reasoning_enabled'], { reasoning_enabled: true }, { reasoning_enabled: false }, 'aesop');
      const lines = readFileSync(path, 'utf-8').trim().split('\n').map((l) => JSON.parse(l));
      expect(lines.map((l) => [l.source, l.from, l.to])).toEqual([['operator_ipc', false, true], ['agent_tool', true, false]]);
      const { statSync } = require('node:fs') as typeof import('node:fs');
      expect(statSync(path).mode & 0o077).toBe(0);
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  });

  test('a failing sink never fails the settings change', async () => {
    const ledger = new SettingsLedger('/x', { write: () => { throw new Error('disk full'); } });
    const errors: unknown[] = [];
    const realError = console.error;
    console.error = (...a: unknown[]) => { errors.push(a); };
    try {
      const mod = new SettingsModule({ ledger });
      await mod.start(fakeCtx().ctx as never);
      const res = operatorSetReasoning(fakeApp(mod, { aesop: 'idle' }), { enabled: true });
      expect(res.ok).toBe(true);
      expect(mod.getReasoning().enabled).toBe(true);
      expect(errors.length).toBeGreaterThan(0); // ...but says so in the host log
    } finally {
      console.error = realError;
    }
  });
});

// ---------------------------------------------------------------------------
// never in the agent's context
// ---------------------------------------------------------------------------

describe('ledger and receipts never enter the agent context', () => {
  test('the module touches only its own state slot, never the context', async () => {
    const { mod, calls } = await startedModule();
    operatorSetReasoning(fakeApp(mod, { aesop: 'idle' }), { enabled: true });
    mod.getAgentSettingsExtension().reset('aesop');
    // fakeCtx exposes nothing else: a context write would have thrown.
    expect(new Set(calls)).toEqual(new Set(['getState', 'setState']));
  });

  test("the agent's agent_settings view carries no ledger or operator provenance", async () => {
    const { mod } = await startedModule();
    operatorSetReasoning(fakeApp(mod, { aesop: 'idle' }), { enabled: true });
    expect(mod.getAgentSettingsExtension().get('aesop')).toEqual({
      reasoning_enabled: true,
      reasoning_budget_tokens: 8192,
      reasoning_display: 'summarized',
    });
  });

  test('the default ledger path sits in DATA_DIR beside llm-calls, not in a workspace mount', () => {
    // Wiring check: index.ts derives the path from DATA_DIR (or the explicit
    // SETTINGS_LEDGER_PATH override); recipes mount workspace paths, never DATA_DIR.
    const src = readFileSync(join(import.meta.dir, '..', 'src', 'index.ts'), 'utf-8');
    expect(src).toContain("process.env.SETTINGS_LEDGER_PATH || join(config.dataDir, 'settings-changes.jsonl')");
  });
});

// ---------------------------------------------------------------------------
// per-call receipt
// ---------------------------------------------------------------------------

describe('per-call thinking receipt in llm-calls', () => {
  const cases: Array<[string, { enabled: boolean; display?: 'summarized' | 'omitted' }, unknown]> = [
    ['absent (reasoning off)', { enabled: false }, null],
    ['enabled, display summarized', { enabled: true, display: 'summarized' }, { type: 'adaptive', display: 'summarized' }],
    ['enabled, display omitted', { enabled: true, display: 'omitted' }, { type: 'adaptive', display: 'omitted' }],
    ['display set while off sends nothing', { enabled: false, display: 'omitted' }, null],
  ];
  for (const [name, r, expected] of cases) {
    test(name, async () => {
      const dir = mkdtempSync(join(tmpdir(), 'arm-receipt-'));
      try {
        const logPath = join(dir, 'llm-calls.jsonl');
        const { agentTurn } = wireAdapter(logPath, () => ({ budgetTokens: 8192, ...r }));
        await agentTurn();
        const line = lastLogLine(logPath);
        expect(line.kind).toBe('stream');
        const summary = line.requestSummary as Record<string, unknown>;
        expect(summary.thinking).toEqual(expected);
        expect(summary.thinkingFrom).toBe('raw');
      } finally {
        rmSync(dir, { recursive: true, force: true });
      }
    });
  }

  test('error before the payload is built falls back to the effective request', async () => {
    const dir = mkdtempSync(join(tmpdir(), 'arm-receipt-'));
    try {
      const logPath = join(dir, 'llm-calls.jsonl');
      const { adapter, agentTurn } = wireAdapter(logPath, () => ({ enabled: true, budgetTokens: 8192, display: 'omitted' }));
      (adapter as unknown as { buildRequest: () => never }).buildRequest = () => { throw new Error('build failed'); };
      await expect(agentTurn()).rejects.toThrow();
      const line = lastLogLine(logPath);
      expect(line.type).toBe('error');
      expect((line.requestSummary as Record<string, unknown>).thinking).toEqual({ type: 'adaptive', display: 'omitted' });
      expect((line.requestSummary as Record<string, unknown>).thinkingFrom).toBe('effective');
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  });

  test('the receipt is metadata only: no thinking text, even when the response has some', async () => {
    const dir = mkdtempSync(join(tmpdir(), 'arm-receipt-'));
    try {
      const logPath = join(dir, 'llm-calls.jsonl');
      const { agentTurn } = wireAdapter(logPath, () => ({ enabled: true, budgetTokens: 8192 }));
      await agentTurn();
      const summary = lastLogLine(logPath).requestSummary as Record<string, unknown>;
      expect(Object.keys(summary.thinking as object).sort()).toEqual(['display', 'type']);
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  });
});

// ---------------------------------------------------------------------------
// the real framework's agent_settings dispatch
// ---------------------------------------------------------------------------

describe("the agent's own agent_settings tool, through the real framework", () => {
  test('update and blanket reset land in the ledger as agent_tool, reasoning and core keys alike', async () => {
    const { AgentFramework } = await import('@animalabs/agent-framework');
    const { Membrane, MockAdapter, NativeFormatter } = await import('@animalabs/membrane');
    const dir = mkdtempSync(join(tmpdir(), 'arm-fw-'));
    const { ledger, lines } = memLedger();
    const settings = new SettingsModule({ ledger });
    const framework = await AgentFramework.create({
      storePath: join(dir, 'store'),
      membrane: new Membrane(new MockAdapter({ defaultResponse: 'ok' }), { formatter: new NativeFormatter() }),
      agents: [{ name: 'aesop', model: 'mock', systemPrompt: 'test', maxTokens: 256 }],
      modules: [settings as never],
    });
    instrumentFrameworkSettings(framework, ledger);
    framework.start();
    try {
      const fw = framework as unknown as {
        dispatchAgentSettingsToolCall(agent: string, call: { id: string; name: string; input: unknown }): void;
      };
      // Operator flip first, then the agent's own update and a blanket reset.
      operatorSetReasoning(
        { framework: framework as never },
        { enabled: true },
      );
      lines.length = 0;
      fw.dispatchAgentSettingsToolCall('aesop', {
        id: 't1', name: 'agent_settings',
        input: { action: 'update', reasoning_display: 'omitted', tail_tokens: 5000 },
      });
      fw.dispatchAgentSettingsToolCall('aesop', { id: 't2', name: 'agent_settings', input: { action: 'reset' } });

      expect(settings.getReasoning().enabled).toBe(false); // the reset undid the flip...
      const flipUndo = lines.find((l) => l.action === 'reset' && l.key === 'reasoning_enabled');
      expect(flipUndo).toMatchObject({ source: 'agent_tool', agent: 'aesop', from: true, to: false }); // ...visibly
      expect(lines.filter((l) => l.action === 'update').map((l) => [l.source, l.key])).toEqual(
        expect.arrayContaining([['agent_tool', 'reasoning_display']]),
      );
      expect(new Set(lines.map((l) => l.source))).toEqual(new Set(['agent_tool']));
      expect(lines.filter((l) => l.action === 'reset').map((l) => l.key)).toEqual(expect.arrayContaining([
        'reasoning_enabled', 'reasoning_budget_tokens', 'reasoning_display',
        'contextBudgetTokens', 'tailTokens', 'transitionPaceTokens', 'sameRoundThinkTextPolicy',
      ]));
    } finally {
      await framework.stop();
      rmSync(dir, { recursive: true, force: true });
    }
  });
});

// ---------------------------------------------------------------------------
// scope: agent turns follow the setting, background calls never get it
// ---------------------------------------------------------------------------

describe('reasoning scope: agent-role calls only', () => {
  for (const display of ['summarized', 'omitted'] as const) {
    test(`a compression (complete) call carries no thinking parameter with reasoning ON, display ${display}`, async () => {
      const dir = mkdtempSync(join(tmpdir(), 'arm-scope-'));
      try {
        const logPath = join(dir, 'llm-calls.jsonl');
        const { sent, background } = wireAdapter(logPath, () => ({ enabled: true, budgetTokens: 8192, display }));
        await background();
        expect(sent[0]!.thinking).toBeUndefined();
        const line = lastLogLine(logPath);
        expect(line.kind).toBe('complete');
        expect(line.requestSummary).toMatchObject({ thinking: null, thinkingFrom: 'raw' });
      } finally {
        rmSync(dir, { recursive: true, force: true });
      }
    });
  }

  test('interleaved: agent turns follow the live setting while compression stays pinned', async () => {
    const dir = mkdtempSync(join(tmpdir(), 'arm-scope-'));
    try {
      const { mod } = await startedModule();
      const app = fakeApp(mod, { aesop: 'idle' });
      const logPath = join(dir, 'llm-calls.jsonl');
      const { sent, agentTurn, background } = wireAdapter(logPath, () => mod.getReasoning());

      await agentTurn();
      await background();
      operatorSetReasoning(app, { enabled: true });
      await agentTurn();
      await background();
      operatorSetReasoning(app, { enabled: false });
      await agentTurn();
      await background();

      expect(sent.map((p) => p.thinking ?? null)).toEqual([
        null, null,
        { type: 'adaptive', display: 'summarized' }, null,
        null, null,
      ]);
      const receipts = readFileSync(logPath, 'utf-8').trim().split('\n')
        .map((l) => JSON.parse(l) as { kind: string; requestSummary: { thinking: unknown } })
        .map((l) => [l.kind, l.requestSummary.thinking]);
      expect(receipts).toEqual([
        ['stream', null], ['complete', null],
        ['stream', { type: 'adaptive', display: 'summarized' }], ['complete', null],
        ['stream', null], ['complete', null],
      ]);
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  });

  test("the agent's tool description says background calls don't use it", () => {
    const desc = (new SettingsModule().getAgentSettingsExtension().properties.reasoning_enabled as { description: string }).description;
    expect(desc).toContain('memory compression');
  });
});
