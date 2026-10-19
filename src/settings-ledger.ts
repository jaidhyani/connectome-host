/**
 * Settings-change ledger: an append-only, host-side JSONL record of every
 * change to an agent setting, whoever made it.
 *
 * Why it exists: an experiment that flips a setting on a sealed schedule
 * (Aesop's intervention arm, 2026-10) has to show at reveal that nothing ELSE
 * moved the setting: not the agent's own `agent_settings` update or blanket
 * reset, not an operator panel, not a restart that loaded an unexpected state.
 * The schedule hash alone can't distinguish "no unlogged lever" from "nobody
 * looked"; this file is the "somebody looked".
 *
 * One line per key a settings call touched:
 *
 *   { t, pid, source, action, agent?, key, from, to }
 *
 * - `source`: who made the call. 'agent_tool' (the agent's own agent_settings
 *   tool), 'operator_panel' (settings-update/-reset/-cancel-transition through
 *   the web UI or the IPC `panel-request` verb), 'operator_ipc' (the
 *   `reasoning-set` IPC verb), 'startup' (the state a module loaded at boot).
 * - `action`: 'update' | 'reset' | 'cancel' | 'set' | 'load'.
 * - A line is written for every key IN SCOPE of the call, including keys whose
 *   value didn't change (`from === to`): a no-op reset is still evidence that
 *   a reset happened. Consumers that want real changes filter `from !== to`.
 *
 * Privacy contract: the ledger is written to a host path (default
 * `{DATA_DIR}/settings-changes.jsonl`, beside llm-calls) and NOTHING in this
 * module touches an agent's context. It holds setting names and values only,
 * never message or thinking text. Writes never throw: a ledger failure must
 * not fail the settings change or the agent's tool call.
 */

import { appendFileSync } from 'node:fs';

export type SettingsChangeSource = 'agent_tool' | 'operator_panel' | 'operator_ipc' | 'startup';
export type SettingsChangeAction = 'update' | 'reset' | 'cancel' | 'set' | 'load';

export interface SettingsChangeLine {
  t: string;
  pid: number;
  source: SettingsChangeSource;
  action: SettingsChangeAction;
  agent?: string;
  key: string;
  from: unknown;
  to: unknown;
}

export interface SettingsLedgerOptions {
  /** Clock override for tests. */
  now?: () => Date;
  /** Line sink override for tests; defaults to appending to `path`. */
  write?: (line: string) => void;
}

export class SettingsLedger {
  readonly path: string;
  private readonly now: () => Date;
  private readonly write: (line: string) => void;

  constructor(path: string, opts: SettingsLedgerOptions = {}) {
    this.path = path;
    this.now = opts.now ?? (() => new Date());
    this.write = opts.write ?? ((line) => appendFileSync(path, line, { mode: 0o600 }));
  }

  /**
   * Record one call's effect. `before`/`after` are flat snapshots; `keys` is
   * the call's scope (every key it was asked to touch). Keys absent from both
   * snapshots are recorded with `null` on the missing side.
   */
  record(
    source: SettingsChangeSource,
    action: SettingsChangeAction,
    keys: readonly string[],
    before: Record<string, unknown> | null,
    after: Record<string, unknown>,
    agent?: string,
  ): void {
    const t = this.now().toISOString();
    for (const key of keys) {
      const line: SettingsChangeLine = {
        t,
        pid: process.pid,
        source,
        action,
        ...(agent !== undefined ? { agent } : {}),
        key,
        from: before ? (before[key] ?? null) : null,
        to: after[key] ?? null,
      };
      try {
        this.write(JSON.stringify(line) + '\n');
      } catch (err) {
        // Never fail the settings change over its audit line; say so loudly
        // in the host log instead, so a dead ledger is discoverable.
        console.error(`[settings-ledger] write failed (${this.path}):`, err);
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Framework core settings (context_budget_tokens & co.)
// ---------------------------------------------------------------------------

/** The framework's runtime-settings keys, in its own (camelCase) names. */
export const CORE_SETTINGS_KEYS = [
  'contextBudgetTokens',
  'tailTokens',
  'transitionPaceTokens',
  'sameRoundThinkTextPolicy',
] as const;

/**
 * Who is calling the framework's runtime-settings methods right now. The
 * framework itself only calls them from its `agent_settings` tool dispatch,
 * so the default is 'agent_tool'; host operator paths wrap their calls in
 * `withSettingsSource('operator_panel', ...)`. The methods are synchronous,
 * so a plain variable with try/finally is an exact scope.
 */
let currentSource: SettingsChangeSource = 'agent_tool';

export function withSettingsSource<T>(source: SettingsChangeSource, fn: () => T): T {
  const prev = currentSource;
  currentSource = source;
  try {
    return fn();
  } finally {
    currentSource = prev;
  }
}

interface RuntimeSettingsFramework {
  getAgentRuntimeSettings(agentName: string): Record<string, unknown>;
  updateAgentRuntimeSettings(agentName: string, patch: Record<string, unknown>, opts?: unknown): unknown;
  resetAgentRuntimeSettings(agentName: string, keys?: string[], opts?: unknown): unknown;
  cancelAgentRuntimeSettingsTransition(agentName: string): unknown;
}

/**
 * Wrap a framework instance's three runtime-settings mutators so every call,
 * from the agent's tool or from an operator path, lands in the ledger. Done
 * on the instance (the framework's tool dispatch calls `this.update…`, which
 * resolves here) so the agent-framework fork needs no change.
 *
 * Feature-detected: a framework build without the runtime-settings surface is
 * left alone.
 */
export function instrumentFrameworkSettings(framework: unknown, ledger: SettingsLedger): void {
  const fw = framework as Partial<RuntimeSettingsFramework> & Record<string, unknown>;
  if (
    typeof fw.getAgentRuntimeSettings !== 'function'
    || typeof fw.updateAgentRuntimeSettings !== 'function'
    || typeof fw.resetAgentRuntimeSettings !== 'function'
    || typeof fw.cancelAgentRuntimeSettingsTransition !== 'function'
  ) return;
  if (fw.__settingsLedgerInstrumented) return;
  fw.__settingsLedgerInstrumented = true;

  const get = fw.getAgentRuntimeSettings.bind(fw);
  const snapshot = (agent: string): Record<string, unknown> | null => {
    try { return get(agent); } catch { return null; }
  };
  const update = fw.updateAgentRuntimeSettings.bind(fw);
  const reset = fw.resetAgentRuntimeSettings.bind(fw);
  const cancel = fw.cancelAgentRuntimeSettingsTransition.bind(fw);

  fw.updateAgentRuntimeSettings = (agent: string, patch: Record<string, unknown>, opts?: unknown) => {
    const before = snapshot(agent);
    const result = update(agent, patch, opts);
    const keys = CORE_SETTINGS_KEYS.filter((k) => patch[k] !== undefined);
    ledger.record(currentSource, 'update', keys, before, snapshot(agent) ?? {}, agent);
    return result;
  };
  fw.resetAgentRuntimeSettings = (agent: string, keys?: string[], opts?: unknown) => {
    const before = snapshot(agent);
    const result = reset(agent, keys, opts);
    const scope = keys && keys.length > 0
      ? CORE_SETTINGS_KEYS.filter((k) => keys.includes(k))
      : CORE_SETTINGS_KEYS;
    ledger.record(currentSource, 'reset', scope, before, snapshot(agent) ?? {}, agent);
    return result;
  };
  fw.cancelAgentRuntimeSettingsTransition = (agent: string) => {
    const before = snapshot(agent);
    const result = cancel(agent);
    const after = snapshot(agent) ?? {};
    // Cancel has no declared key scope: it settles the budget where the
    // transition stands. Record the keys it actually moved, plus the
    // transition state, so a cancel is never invisible.
    const moved = CORE_SETTINGS_KEYS.filter((k) => before?.[k] !== after[k]);
    ledger.record(currentSource, 'cancel', [...moved, 'transition'], before, after, agent);
    return result;
  };
}
