/**
 * SettingsModule — runtime-tunable host settings the agent can toggle for
 * itself. State persists to chronicle via ModuleContext (`setState`/`getState`)
 * so changes survive restarts.
 *
 * First domain: **reasoning** (Anthropic extended thinking), surfaced to the
 * agent as `agent_settings` fields (reasoning_enabled /
 * reasoning_budget_tokens) via the framework's settings-extension hook —
 * NOT as standalone tools (the former reasoning_status/enable/disable trio
 * was tool bloat for one boolean + number).
 *
 * The host's adapter wrapper (LoggingAnthropicAdapter) reads `getReasoning()`
 * on each call and injects `thinking: {type:'enabled', budget_tokens: N}` into
 * the outgoing Anthropic request when enabled — keeping the cross-cutting
 * "request mutator" plumbing out of every call site.
 *
 * Designed to be extensible: new domains add their own slice in
 * `SettingsState` + a few tools + a typed accessor. Bundled with the host;
 * recipes opt in by including `SettingsModule` in moduleInstances.
 */

import type {
  Module,
  ModuleContext,
  ProcessEvent,
  ProcessState,
  EventResponse,
  ToolDefinition,
  ToolCall,
  ToolResult,
} from '@animalabs/agent-framework';
import type { SettingsLedger, SettingsChangeSource } from '../settings-ledger.js';

export interface ReasoningSettings {
  enabled: boolean;
  budgetTokens: number;
  /**
   * How thinking content comes back from the API: 'summarized' returns a
   * readable reasoning summary in the `thinking` field; 'omitted' returns an
   * empty `thinking` field with only the encrypted signature. Models 4.7+
   * default to 'omitted' server-side — we default to 'summarized' to restore
   * the pre-4.7 behavior (visible reasoning in stores, webui, estimators).
   */
  display: 'summarized' | 'omitted';
}

export interface SettingsState {
  reasoning: ReasoningSettings;
}

const DEFAULTS: SettingsState = {
  reasoning: { enabled: false, budgetTokens: 8192, display: 'summarized' },
};

/** The reasoning keys, in their agent_settings (wire) names. */
const REASONING_KEYS = ['reasoning_enabled', 'reasoning_budget_tokens', 'reasoning_display'] as const;

/** What an operator may set from outside the agent (the `reasoning-set` IPC
 *  verb): the two levers, nothing else. Budget is informational under
 *  adaptive thinking and is never sent, so it isn't a lever. */
export interface OperatorReasoningPatch {
  enabled?: boolean;
  display?: 'summarized' | 'omitted';
}

export class SettingsModule implements Module {
  readonly name = 'settings';

  private ctx: ModuleContext | null = null;
  private state: SettingsState = clone(DEFAULTS);
  private readonly ledger?: SettingsLedger;

  /** `ledger`: where every reasoning change is recorded, whoever makes it
   *  (see src/settings-ledger.ts). Optional so tests and other hosts can
   *  construct the module bare. */
  constructor(opts: { ledger?: SettingsLedger } = {}) {
    this.ledger = opts.ledger;
  }

  async start(ctx: ModuleContext): Promise<void> {
    this.ctx = ctx;
    const saved = ctx.getState<Partial<SettingsState>>();
    if (saved) {
      // Shallow-merge each domain so future-added fields fall back to defaults
      // for state persisted by older versions.
      this.state = {
        reasoning: { ...DEFAULTS.reasoning, ...(saved.reasoning ?? {}) },
      };
    }
    // What this process will run with, recorded at every boot: a restart that
    // loaded a flipped (or unexpectedly reverted) state is visible in the
    // ledger rather than inferred.
    this.ledger?.record('startup', 'load', REASONING_KEYS, null, this.reasoningSettingsView());
  }

  async stop(): Promise<void> {
    this.ctx = null;
  }

  /** Read accessor for external consumers (e.g., the LLM adapter wrapper). */
  getReasoning(): ReasoningSettings {
    return { ...this.state.reasoning };
  }

  /**
   * Operator write path for the reasoning levers, used by the headless IPC
   * `reasoning-set` verb. Same effect as the agent's own agent_settings update
   * (in-memory state the adapter reads on the next call, persisted to module
   * state so a restart keeps it), with three differences that are the point:
   * it accepts only `enabled` and `display`; it is recorded in the ledger as
   * `operator_ipc`; and it never tells the agent anything. Whether the agent
   * is mid-turn is the caller's check (src/operator-reasoning.ts).
   */
  operatorSetReasoning(patch: OperatorReasoningPatch): {
    before: Record<string, unknown>;
    after: Record<string, unknown>;
  } {
    const unknown = Object.keys(patch).filter((k) => k !== 'enabled' && k !== 'display');
    if (unknown.length > 0) {
      throw new Error(`reasoning-set accepts only enabled and display (got ${unknown.join(', ')})`);
    }
    if (patch.enabled === undefined && patch.display === undefined) {
      throw new Error('reasoning-set needs enabled and/or display');
    }
    const wire: Record<string, unknown> = {};
    if (patch.enabled !== undefined) wire.reasoning_enabled = patch.enabled;
    if (patch.display !== undefined) wire.reasoning_display = patch.display;
    const before = this.reasoningSettingsView();
    this.applyReasoningPatch(wire, 'operator_ipc', 'set');
    return { before, after: this.reasoningSettingsView() };
  }

  /** Validate and apply a wire-named reasoning patch, persist it, and record
   *  it. Throws (applying nothing) on any invalid field. */
  private applyReasoningPatch(
    patch: Record<string, unknown>,
    source: SettingsChangeSource,
    action: 'update' | 'set',
    agent?: string,
  ): void {
    const next = { ...this.state.reasoning };
    if (patch.reasoning_enabled !== undefined) {
      if (typeof patch.reasoning_enabled !== 'boolean') {
        throw new Error('reasoning_enabled must be a boolean');
      }
      next.enabled = patch.reasoning_enabled;
    }
    if (patch.reasoning_budget_tokens !== undefined) {
      const budget = Number(patch.reasoning_budget_tokens);
      if (!Number.isFinite(budget)) {
        throw new Error('reasoning_budget_tokens must be a number');
      }
      next.budgetTokens = Math.max(1024, Math.round(budget));
    }
    if (patch.reasoning_display !== undefined) {
      if (patch.reasoning_display !== 'summarized' && patch.reasoning_display !== 'omitted') {
        throw new Error("reasoning_display must be 'summarized' or 'omitted'");
      }
      next.display = patch.reasoning_display;
    }
    const before = this.reasoningSettingsView();
    this.state.reasoning = next;
    this.ctx?.setState(this.state);
    const keys = REASONING_KEYS.filter((k) => patch[k] !== undefined);
    this.ledger?.record(source, action, keys, before, this.reasoningSettingsView(), agent);
  }

  /** No standalone tools — reasoning controls live inside the framework's
   *  `agent_settings` tool via getAgentSettingsExtension() below. The three
   *  former reasoning_* tools were pure tool bloat for one boolean + number. */
  getTools(): ToolDefinition[] {
    return [];
  }

  async handleToolCall(call: ToolCall): Promise<ToolResult> {
    return {
      success: false,
      error:
        `Unknown tool: ${call.name}. Reasoning controls moved into agent_settings ` +
        `(fields reasoning_enabled / reasoning_budget_tokens).`,
      isError: true,
    };
  }

  /**
   * Declare reasoning as an agent_settings extension: the framework merges
   * these fields into the agent_settings tool and routes get/update/reset for
   * them back here. Framework versions predating the hook simply never call
   * this — in that case reasoning is temporarily not agent-tunable (the
   * adapter still honors persisted state).
   */
  getAgentSettingsExtension(): {
    properties: Record<string, unknown>;
    keys: string[];
    get(agentName: string): Record<string, unknown>;
    update(agentName: string, patch: Record<string, unknown>): Record<string, unknown>;
    reset(agentName: string, keys?: string[]): Record<string, unknown>;
  } {
    return {
      properties: {
        reasoning_enabled: {
          type: 'boolean',
          description:
            'Extended thinking (reasoning) on subsequent inference calls.',
        },
        reasoning_budget_tokens: {
          type: 'number',
          description: 'Token budget for thinking blocks (min 1024).',
        },
        reasoning_display: {
          type: 'string',
          enum: ['summarized', 'omitted'],
          description:
            "How your thinking is returned: 'summarized' (a readable summary of your reasoning " +
            "is recorded alongside the signature) or 'omitted' (signature only, slightly faster " +
            'first token; your reasoning is not visible to anyone, including you on replay).',
        },
      },
      keys: ['reasoning_enabled', 'reasoning_budget_tokens', 'reasoning_display'],
      get: () => this.reasoningSettingsView(),
      update: (agentName, patch) => {
        this.applyReasoningPatch(patch, 'agent_tool', 'update', agentName);
        return this.reasoningSettingsView();
      },
      // A blanket `agent_settings reset` (no `settings` list) lands here with
      // no keys and returns ALL reasoning keys to DEFAULTS, which silently
      // undoes an operator flip. Its semantics are deliberately unchanged
      // (changing them would itself be a tell); instead every reset is
      // recorded, per key, so the experiment can void the trials it touches.
      reset: (agentName, keys) => {
        const before = this.reasoningSettingsView();
        const all = !keys || keys.length === 0;
        if (all || keys?.includes('reasoning_enabled')) {
          this.state.reasoning.enabled = DEFAULTS.reasoning.enabled;
        }
        if (all || keys?.includes('reasoning_budget_tokens')) {
          this.state.reasoning.budgetTokens = DEFAULTS.reasoning.budgetTokens;
        }
        if (all || keys?.includes('reasoning_display')) {
          this.state.reasoning.display = DEFAULTS.reasoning.display;
        }
        this.ctx?.setState(this.state);
        const scope = all ? REASONING_KEYS : REASONING_KEYS.filter((k) => keys?.includes(k));
        this.ledger?.record('agent_tool', 'reset', scope, before, this.reasoningSettingsView(), agentName);
        return this.reasoningSettingsView();
      },
    };
  }

  /** The extension's wire view of reasoning state (flat agent_settings keys). */
  private reasoningSettingsView(): Record<string, unknown> {
    return {
      reasoning_enabled: this.state.reasoning.enabled,
      reasoning_budget_tokens: this.state.reasoning.budgetTokens,
      reasoning_display: this.state.reasoning.display,
    };
  }

  async onProcess(_event: ProcessEvent, _state: ProcessState): Promise<EventResponse> {
    return {};
  }

}

function clone<T>(x: T): T {
  return JSON.parse(JSON.stringify(x)) as T;
}
