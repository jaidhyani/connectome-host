/**
 * Operator control of the reasoning levers over the headless IPC socket.
 *
 * Two verbs (wired in src/headless.ts):
 *
 *   {type:"reasoning-get", corrId?}
 *     → {type:"reasoning-get-result", corrId, ok:true, reasoning, idle, statuses}
 *
 *   {type:"reasoning-set", corrId?, enabled?: boolean, display?: "summarized"|"omitted"}
 *     → {type:"reasoning-set-result", corrId, ok:true, before, after}
 *     → {type:"reasoning-set-result", corrId, ok:false, error:"busy", statuses}
 *     → {type:"reasoning-set-result", corrId, ok:false, error:"<validation message>"}
 *
 * Built for an experiment in which an operator flips an agent's
 * `reasoning_enabled` on a schedule the agent doesn't see (Aesop's
 * intervention arm, 2026-10). The change takes effect on the next model call:
 * the adapter reads SettingsModule.getReasoning() per call. No restart.
 *
 * Mid-turn: REFUSED, not queued. A set is accepted only when every agent in
 * the process is `idle`, so a flip never lands between rounds of one tool
 * loop (where turning thinking on can draw an API error or be dropped), and
 * the moment it applied is the moment the operator asked. A queued flip
 * would apply at a time the caller doesn't know; a refusal lets the caller
 * record and retry. The check and the write run in one synchronous step, so
 * no inference can start between them.
 *
 * The agent is never told: nothing here writes to a context manager or emits
 * to anything but the requesting socket client. The change is recorded in the
 * host-side settings ledger as source `operator_ipc`.
 */

import type { OperatorReasoningPatch, SettingsModule } from './modules/settings-module.js';

export interface OperatorReasoningApp {
  framework: {
    getAllModules(): Array<{ name: string }>;
    getAllAgents(): Array<{ name: string; state: { status: string } }>;
  };
}

export type OperatorReasoningResult =
  | { ok: true; before: Record<string, unknown>; after: Record<string, unknown> }
  | { ok: false; error: string; statuses?: Record<string, string> };

function findSettings(app: OperatorReasoningApp): SettingsModule | undefined {
  return app.framework.getAllModules().find((m) => m.name === 'settings') as SettingsModule | undefined;
}

function agentStatuses(app: OperatorReasoningApp): Record<string, string> {
  const out: Record<string, string> = {};
  for (const a of app.framework.getAllAgents()) out[a.name] = a.state.status;
  return out;
}

export function operatorGetReasoning(app: OperatorReasoningApp):
  | { ok: true; reasoning: Record<string, unknown>; idle: boolean; statuses: Record<string, string> }
  | { ok: false; error: string } {
  const settings = findSettings(app);
  if (!settings) return { ok: false, error: 'settings module not loaded' };
  const r = settings.getReasoning();
  const statuses = agentStatuses(app);
  return {
    ok: true,
    reasoning: { reasoning_enabled: r.enabled, reasoning_display: r.display },
    idle: Object.values(statuses).every((s) => s === 'idle'),
    statuses,
  };
}

export function operatorSetReasoning(
  app: OperatorReasoningApp,
  params: Record<string, unknown>,
): OperatorReasoningResult {
  const settings = findSettings(app);
  if (!settings) return { ok: false, error: 'settings module not loaded' };

  const patch: OperatorReasoningPatch & Record<string, unknown> = {};
  for (const [k, v] of Object.entries(params)) {
    if (k === 'type' || k === 'corrId') continue;
    patch[k] = v;
  }
  if (patch.enabled !== undefined && typeof patch.enabled !== 'boolean') {
    return { ok: false, error: 'enabled must be a boolean' };
  }
  if (patch.display !== undefined && patch.display !== 'summarized' && patch.display !== 'omitted') {
    return { ok: false, error: "display must be 'summarized' or 'omitted'" };
  }

  const statuses = agentStatuses(app);
  if (!Object.values(statuses).every((s) => s === 'idle')) {
    return { ok: false, error: 'busy', statuses };
  }
  try {
    return { ok: true, ...settings.operatorSetReasoning(patch) };
  } catch (err) {
    return { ok: false, error: err instanceof Error ? err.message : String(err) };
  }
}
