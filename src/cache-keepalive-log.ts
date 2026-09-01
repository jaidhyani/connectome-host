// Where prompt-cache keepalive events get written.
//
// Every event goes to STDERR, deliberately and without exception.
//
// The host's systemd unit routes `StandardError` to
// `<data>/service-stderr.log` and leaves stdout on the journal. That file is
// where an operator actually looks — it is where `[inference-refusal]` and the
// `[autobiographical]` lines live. Splitting keepalive events across two sinks
// by severity means the routine ones land somewhere nobody greps.
//
// This is not hypothetical. On 2026-08-23 the keepalive ran correctly on
// fable-cm for three hours — three clean refreshes, 523,102 tokens read each,
// zero writes — while a monitor tailing service-stderr.log reported
// `keepalive=0` the entire time, because `refreshed` was going to stdout. An
// operator would have concluded the feature was dead. A background spender
// that cannot be found in the log an operator reads is indistinguishable from
// one that never ran.
//
// Volume does not justify splitting them: one refresh per lineage per ~50
// minutes is nothing next to the compression chatter already in that file.

import type { KeepaliveEvent } from '@animalabs/membrane';

export const KEEPALIVE_LOG_PREFIX = '[cache-keepalive]';

/** Render one event as a single log line. */
export function formatKeepaliveEvent(event: KeepaliveEvent): string {
  return `${KEEPALIVE_LOG_PREFIX} ${JSON.stringify(event)}`;
}

/**
 * Write one keepalive event to the operator-visible log.
 *
 * `sink` exists for tests; production always uses stderr via the default.
 */
export function logKeepaliveEvent(
  event: KeepaliveEvent,
  sink: (line: string) => void = (line) => console.error(line),
): void {
  sink(formatKeepaliveEvent(event));
}
