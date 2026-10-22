import { expect, test } from 'vitest'

import { CanonicalDesktopProtocol } from './canonical-protocol'

// A typed /undo or /retry derives its target from a fresh snapshot. After a lost reply the owner has
// already rewound, so the repeat's snapshot names the turn before: the retained rewind, not a new one at
// that target, must be what the repeat sends (as Ink's mutateCanonical and the classic CLI do).
function primed() {
  const protocol = new CanonicalDesktopProtocol()
  protocol.result('session.resume', { session_id: 's' }, { session_id: 's', revision: 7, execution_generation: 3 })

  return protocol
}

const derived = (target: number) => ({ session_id: 's', target_message_id: target, derived_target: true })

test('a derived rewind whose reply was lost is re-sent for the repeat, not re-derived', () => {
  const protocol = primed()
  const lost = protocol.prepare('session.rewind', derived(5))
  protocol.failure(lost, new Error('Hermes gateway connection lost'))

  // The repeat's snapshot already lacks row 5 and names row 3; the original request goes out again.
  protocol.result('session.resume', { session_id: 's' }, { session_id: 's', revision: 8, execution_generation: 4 })
  const repeat = protocol.prepare('session.rewind', derived(3))

  expect(repeat).toEqual(lost)
  expect(repeat).toMatchObject({ payload: { target_message_id: 5 }, expected_revision: 7, expected_generation: 3 })
})

test('an answered derived rewind is not replayed for the next /undo', () => {
  const protocol = primed()
  const first = protocol.prepare('session.rewind', derived(5))
  protocol.result('session.rewind', first, {
    session_id: 's',
    revision: 8,
    rewound_count: 2,
    target_message: { id: 5 }
  })

  const next = protocol.prepare('session.rewind', derived(3))

  expect(next.request_id).not.toBe(first.request_id)
  expect(next).toMatchObject({ payload: { target_message_id: 3 }, expected_revision: 8 })
})

test('a typed refusal of a derived rewind lets the repeat derive again', () => {
  const protocol = primed()
  const refused = protocol.prepare('session.rewind', derived(5))
  protocol.failure(refused, { data: { reason: 'revision_conflict', revision: 9 } })

  const retry = protocol.prepare('session.rewind', derived(5))

  expect(retry.request_id).not.toBe(refused.request_id)
  expect(retry).toMatchObject({ payload: { target_message_id: 5 }, expected_revision: 9 })
})

test('another control retires a lost derived rewind', () => {
  const protocol = primed()
  const lost = protocol.prepare('session.rewind', derived(5))
  protocol.failure(lost, new Error('socket closed'))

  protocol.prepare('session.title', { session_id: 's', title: 'renamed' })
  const later = protocol.prepare('session.rewind', derived(3))

  expect(later.request_id).not.toBe(lost.request_id)
  expect(later).toMatchObject({ payload: { target_message_id: 3 } })
})

test('an explicit rewind (edit, regenerate, restore) stays keyed by its own target', () => {
  const protocol = primed()
  const lost = protocol.prepare('session.rewind', { session_id: 's', target_message_id: 5 })
  protocol.failure(lost, new Error('socket closed'))

  expect(protocol.prepare('session.rewind', { session_id: 's', target_message_id: 5 })).toEqual(lost)

  const other = protocol.prepare('session.rewind', { session_id: 's', target_message_id: 3 })

  expect(other.request_id).not.toBe(lost.request_id)
  expect(other).toMatchObject({ payload: { target_message_id: 3 } })
})

// A retained rewind is keyed by the SESSION, not by the command that asked for it: the /undo's
// receipt is what the next /retry replays (as Ink's mutateCanonical does), and another session's
// own rewind is never handed that receipt.
test('a lost rewind is replayed by the next rewind of the same session, whatever the target', () => {
  const protocol = primed()
  const lost = protocol.prepare('session.rewind', derived(14))
  protocol.failure(lost, new Error('Hermes gateway connection lost'))

  // The /undo's reply is gone: the owner already rewound, so the transcript a new target would
  // come from may hold no turn this command accepts. The retained rewind goes out again.
  const retry = protocol.prepare('session.rewind', derived(11))

  expect(retry).toEqual(lost)
  expect(retry).toMatchObject({ payload: { target_message_id: 14 }, expected_revision: 7, expected_generation: 3 })
})

test('a retained rewind belongs to its session: another session derives its own target', () => {
  const protocol = primed()
  const lost = protocol.prepare('session.rewind', derived(14))
  protocol.failure(lost, new Error('Hermes gateway connection lost'))

  // The other session's rewind owes nothing of the lost one: it has its own revision and
  // generation, and its own target.
  protocol.result('session.resume', { session_id: 'other' }, { session_id: 'other', revision: 2, execution_generation: 1 })
  const other = protocol.prepare('session.rewind', { session_id: 'other', target_message_id: 3, derived_target: true })

  expect(other.request_id).not.toBe(lost.request_id)
  expect(other).toMatchObject({ payload: { target_message_id: 3 }, expected_revision: 2, expected_generation: 1 })

  // The replay-only probe of that session re-presents that session's own rewind, never the lost one.
  const probe = protocol.prepare('session.rewind', {
    session_id: 'other', derived_target: true, replay_only: true
  })

  expect(probe).toEqual(other)
  expect(probe.request_id).not.toBe(lost.request_id)
})

test('an owner refusal that names a reason lets the next /undo derive again', () => {
  const protocol = primed()
  const refused = protocol.prepare('session.rewind', derived(5))
  protocol.failure(refused, { data: { reason: 'invalid_params', field: 'target_message_id' } })

  const next = protocol.prepare('session.rewind', derived(3))

  expect(next.request_id).not.toBe(refused.request_id)
  expect(next).toMatchObject({ payload: { target_message_id: 3 }, expected_revision: 7 })
})

test('a replay-only probe re-presents only a retained derived rewind', () => {
  const protocol = primed()
  const probe = { session_id: 's', derived_target: true, replay_only: true }

  expect(protocol.localAnswer('session.rewind', probe)).toMatchObject({ status: 'nothing', rewound_count: 0 })
  expect(() => protocol.prepare('session.rewind', probe)).toThrow('No retained rewind')

  const lost = protocol.prepare('session.rewind', derived(5))
  protocol.failure(lost, new Error('socket closed'))

  expect(protocol.localAnswer('session.rewind', probe)).toBeUndefined()
  expect(protocol.prepare('session.rewind', probe)).toEqual(lost)
})
