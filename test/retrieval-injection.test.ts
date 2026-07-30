import { describe, expect, test } from 'bun:test';
import { RetrievalModule } from '../src/modules/retrieval-module';
import type { Lesson } from '../src/modules/lessons-module';

// Contract tests for where retrieved lessons land in the compiled context.
//
// Position matters for prompt caching: a `system`-position injection is
// appended to the system prompt, so every selection change rewrites the
// earliest bytes of the request and invalidates the entire cached prefix
// (measured on live traffic at ~25x the warm-call cache-write volume).
// `afterUser` rides the always-uncached tail instead; membrane's
// mergeConsecutiveRoles folds it into the newest user envelope on the wire.
//
// The heading is part of the contract too: once the block stops being
// system-voiced, un-attributed text reads as third-party injection and is
// occasionally explicitly distrusted; the self-attributing heading keeps
// both facts (whose words, how surfaced) on the label.

const LESSON: Lesson = {
  id: 'abc12345',
  content: 'When sourdough comes up, remember: the levain peaks at 3 hours.',
  confidence: 0.9,
  tags: ['sourdough'],
  evidence: [],
  created: 1,
  updated: 1,
  deprecated: false,
};

function makeModule(lessons: Lesson[]) {
  // Step-1 stub: Haiku flags "sourdough" as the concept needing background.
  // With <=3 candidates the validation step is skipped, so one stubbed
  // complete() covers the whole pipeline.
  const membrane = {
    complete: async () => ({
      content: [{ type: 'text', text: '["sourdough"]' }],
    }),
  };
  const mod = new RetrievalModule({ membrane: membrane as never });
  const ctx = {
    getModule: (name: string) =>
      name === 'lessons' ? { getLessons: () => lessons } : null,
    queryMessages: () => ({
      messages: [
        {
          participant: 'user',
          content: [
            { type: 'text', text: 'How long should the levain sit before it peaks?' },
          ],
        },
      ],
    }),
  };
  return { mod, ctx: ctx as never };
}

describe('RetrievalModule injection contract', () => {
  test('retrieved lessons inject at afterUser, not into the system prompt', async () => {
    const { mod, ctx } = makeModule([LESSON]);
    await mod.start(ctx);
    const injections = await mod.gatherContext('agent');
    expect(injections).toHaveLength(1);
    expect(injections[0].namespace).toBe('retrieval');
    expect(injections[0].position).toBe('afterUser');
  });

  test('heading self-attributes with mechanical provenance visible', async () => {
    const { mod, ctx } = makeModule([LESSON]);
    await mod.start(ctx);
    const [injection] = await mod.gatherContext('agent');
    const block = injection.content[0];
    if (block.type !== 'text') throw new Error('expected a text block');
    expect(block.text).toContain('## Retrieved Knowledge');
    expect(block.text).toContain('your own lessons');
    expect(block.text).toContain('machine-surfaced');
    expect(block.text).toContain('- [90%] When sourdough comes up');
  });

  test('no injection when flagged concepts match no lesson', async () => {
    const { mod, ctx } = makeModule([
      { ...LESSON, id: 'def67890', content: 'Pin base images to the registry mirror.', tags: ['docker'] },
    ]);
    await mod.start(ctx);
    expect(await mod.gatherContext('agent')).toHaveLength(0);
  });
});
