# What this sweep does and does not prove

Written 2026-09-15 23:58, while the run was at 150/240 cells. Anything claimed from
`results.jsonl` has to be read against this list. Each item is a limit of the experiment,
not a bug to fix before reporting.

## 1. The context axis stops at 8192, not 32768

Vidur does not simulate attention, it predicts it from a random forest fitted to profiled
data. For this model the profile's `kv_cache_size` column tops out at **16,320 tokens**.
The sweep's `MAX_TOKENS` was set to 32,768.

So every `prefill=16384` cell is **extrapolated, not simulated**: 16384 at pd=32 needs
16,912 tokens and at pd=4 needs 20,496, both past the fitted range. That is **48 of 240
cells**. Everything else in the grid tops out at 10,256 tokens and is inside the profile.

The tell is visible in the output. TBT p99 at prefill 8192, pd=4 reads 0.170 / 0.170 /
0.169, flat across repeats. At 16384, pd=4 it reads 0.997 / 0.997 / 0.934. A six-fold jump
at exactly the rung where the forest leaves its training data.

**Report the context axis as 512 → 8192.** Name 16384 as excluded, and say why: the ceiling
on this experiment is the profile, not the simulator. 192 cells still answer the question.
Extending the axis honestly means profiling the model at longer context first, not raising
the constant. `MAX_TOKENS` is part of the predictor cache key, so raising it also discards
every fit.

## 2. p99 over 128 requests is barely a p99

`synthetic_request_generator_config_num_requests` is **128**. The p99 index into 128 sorted
values is element 126 or 127, so "p99" here is the second-worst or the worst request. It
moves a lot between repeats for that reason alone.

Use **p50 for the crossover claim** and treat p99 as directional. If a tail claim matters,
rerun the interesting cells at 1000+ requests rather than arguing from these.

## 3. Every request is the same size, so there is no queueing variance

The sweep uses `length_generator_config_type = fixed`: identical prefill and decode for all
128 requests in a cell. Real serving is heavy-tailed, and batching behaves very differently
when a long request blocks a batch of short ones — head-of-line blocking is precisely what
TP/PP choice interacts with.

This makes the sweep a **clean comparison between parallelism configs**, and *not* a
prediction of production latency. Say so. The honest framing is "at fixed request shape,
the TP/PP crossover sits at X", with a named follow-up: rerun the crossover region with a
`trace` or heavy-tailed generator.

## 4. makespan_s is null in results.jsonl

`collect()` in `sweep.py` reads `request_completion_time`, which is not a column Vidur
emits, so the key is null on every row. Not corrupt, just absent.

Backfilled instead by **`makespan.py`**, which derives it from data already on disk:
arrival_i is the running sum of `request_inter_arrival_delay` (blank first row = t0), and
makespan = max(arrival_i + `request_e2e_time`). Output: `makespan.jsonl`, one line per
complete cell, joinable on the run directory.

`sweep.py` itself was **left unedited on purpose** — it was mid-run and its loky workers
re-import the module. Fix it there after the run finishes.

---

## Addendum, 2026-09-19 (written after the run finished)

## 5. TP8/PP1 cannot run on pairwise NVLink at all

All 30 of its cells fail with `Training data for model all_reduce is empty`. This is not
a gap in the profiling data to be filled later: an 8-way all-reduce needs the NVSwitch
fabric, and pairwise NVLink only wires GPUs in twos.

The consequence for analysis is that **the two topologies do not have the same field**.
DGX offers four splits, pairwise offers three. Computing winners pooled across both
networks compares a 4-way contest against a 3-way one and produces a cleaner, wronger
answer. `analyse.py` computes winners per network and refuses to pool.

## Items 4 fixed, and a second bug found while fixing it

- `collect()` in `sweep.py` now derives makespan from inter-arrival gaps instead of
  reading the non-existent `request_completion_time`. Verified against `makespan.py`
  across all 222 cells that existed at the time: exact match on 210, and the 12
  disagreements were the second bug.
- **`makespan.py` reported 0.0 s for 12 cells in which no request ever completed.**
  At prefill >= 2048 with pd=0.25, generation runs past the simulated horizon, so all
  128 rows have a blank `request_e2e_time`. The accumulator started at `0.0`, so
  "nothing finished" came out looking like the fastest configuration on the board.
  Both scripts now return `None`. `makespan.jsonl` regenerated: 282 cells, 12 null.
