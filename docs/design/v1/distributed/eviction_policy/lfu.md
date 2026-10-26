# MP LFU eviction policy

The optional `LFU` policy protects KV chunks that are accessed repeatedly. It
is intended for MP deployments whose workload has stable hot prefixes that can
be displaced by a scan under pure LRU.

Each admitted key starts with frequency one. An explicit cache access increases
the frequency, and eviction selects the lowest-frequency keys first. Keys with
the same frequency are ordered by recency, so the least recently accessed key
is selected first. Batch admission preserves the existing MP prefix behavior:
later suffix chunks are candidates before earlier prefix chunks.

The MP L1 write lifecycle emits both reservation and completion events. LFU
records a pending admission so those two events count as one use. Explicit
touches remain the only way to increase a newly admitted key beyond frequency
one.

Select the policy with:

```console
lmcache_server --l1-size-gb 8 --eviction-policy LFU
```

`LFU` is global rather than `cache_salt` isolated and counts keys rather than
bytes. `LRU` remains the default behavior; operators should choose `LFU` only
when reuse frequency is a better predictor of future hits than recent access.
