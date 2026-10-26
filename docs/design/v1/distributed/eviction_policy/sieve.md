# SIEVE eviction policy

`SIEVE` is an opt-in eviction policy for multiprocess L1 and L2 cache tiers. It
keeps keys in FIFO insertion order and stores one visited bit per key. A cache
access sets that bit without changing queue order. Under memory pressure, a
persistent hand scans the queue: visited entries have their bit cleared and
receive one second chance, while unvisited entries become eviction candidates.

```text
touch(B)                         eviction scan
   |                                  |
   v                                  v
[A:0] [B:1] [C:0]  --hand-->  A evicted, B cleared, C evicted
```

This design keeps the hit path at an in-place boolean update rather than an LRU
list move. It also prevents a one-time sequential scan from immediately
displacing an entry that was reused before the eviction cycle. Selection is
still key-count based, matching the existing MP eviction policy contract; it is
not byte-weighted for variable-size objects.

## MP lifecycle

The existing `L1EvictionPolicy` and `L2EvictionPolicy` listeners feed SIEVE the
same lifecycle notifications as other policies. Write reservations enter the
queue as cold keys, completed writes preserve that state, explicit accesses set
the visited bit, and confirmed removals delete queue state. The hand advances
across eviction cycles, including cycles where locked keys are skipped.

Keys from one request are inserted in reverse order. This preserves the MP
prefix-cache convention that suffix chunks should be considered before prefix
chunks, because dropping an early prefix also makes its suffixes unusable.

## Configuration

SIEVE is explicit and does not change the current default:

```bash
lmcache server --l1-size-gb 20 --eviction-policy SIEVE
```

L2 adapters use the same value in their eviction block:

```json
{
  "eviction": {
    "eviction_policy": "SIEVE",
    "trigger_watermark": 0.8,
    "eviction_ratio": 0.2
  }
}
```

SIEVE is global rather than `cache_salt` isolated. Deployments that require
per-tenant quotas should continue to use `IsolatedLRU`.
