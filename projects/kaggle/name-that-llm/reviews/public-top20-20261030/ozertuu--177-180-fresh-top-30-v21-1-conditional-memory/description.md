# 177/180 Fresh Top-30 | v21.1 Conditional Memory

**A public-state nearest-route memory that changes order, not inventory.**

The old public v21 lost its route edge as the leaderboard moved. On a newer
Top-30 snapshot, its Dennis route fell to **19/46**. A fresh current route plus
an identity-free conditional order controller reached:

| Gate | Result | Seat 0 | Seat 1 |
|---|---:|---:|---:|
| reserved market-option validation | **53/53** | 28/28 | 25/25 |
| unopened balanced outer | **59/60** | 30/30 | 29/30 |
| unopened full outer | **177/180** | 85/85 | 92/95 |
| collected only after the data cutoff | **46/51** | 20/23 | 26/28 |

`177/180` is a local counterfactual replay holdout, **not an official Public-LB
score**. The 180 rows are six newest trajectories for each of 30 scoring
submissions; 134 are unique episodes. The artifact was frozen before that
outer slice was opened.

```text
complete current route
  + 30 fit-only public route memories
  + nearest public-farm match at the current step
  + intersection with SELLs already chosen by our route
  + order-only front-loading (no new SELL, no quantity change)
  + bounded WEED repair and legal terminal liquidation
```

Runtime excludes team names, team/submission ids, episode ids, seed, Notebook
ownership, opponent private inventory and hidden current actions.


## 1. Diagnosis: the route moved before the controller did

The previous v21 artifact still won **65/67**
recorded live games under its best small timing ablation, but that statistic hid
the population shift. Against the newer route-validation window:

| Complete route, no learned market intervention | Wins |
|---|---:|
| old public v21 / Dennis medoid | **19/46** |
| current Konstantin medoid | **40/46** |
| current Richard Silence medoid | **41/46** |
| current `안녕하세요...` medoid | **41/46** |

The first-order repair is therefore a route refresh, not a more aggressive
threshold. The top three current families are close: Richard and Konstantin
differ at only 110 board/market channel moments, while both are more than 1,200
channel differences away from the old Dennis route.

The four recorded v21 losses also explained the close-score symptom: margins
were `-41`, `-357`, `-2,671`, and `-3,378`. Moving a quarter-batch two steps
earlier rescued two of four, but it was not promoted because the fresh route
already contained newer market timing. On that route, unconditional early
SELLs reduced rather than improved validation wins.


## 2. Chronological protocol: three development windows, then one outer

Snapshot cutoff: `2026-08-06T02:52:14.073233+00:00`. The download contained
**515 unique episodes**
from 30 current scoring submissions.

For every submission, episodes were ordered newest first and split as follows:

```text
newest 6  -> outer (never loaded during selection)
next 3    -> complete market-option validation
next 3    -> complete route validation
older 12  -> fit-only route donors / public-state prototypes
```

One real 719-action medoid per submission was allowed into memory, preventing
replay-heavy teams from dominating. Route and option selection used disjoint
windows. After freezing SHA-256 `d9dc24ce5429ec62...`, the newest-six outer
was opened once. A second current-leaderboard download then supplied 51
team/seat-balanced, post-cutoff cases—including newly arrived Ben Hamilton,
Subin An, OceanMix, and other submissions not present in the first Top-30.

Replay opponents are fixed public trajectories and do not react to our
counterfactual actions. Results are therefore policy-screening evidence, not a
claim of exact live-LB reproduction.


## 3. Public research lineage and credit boundary

| Public work | Mechanism used or tested here |
|---|---|
| [Kaito v20 — 159/160 vs Frontier](https://www.kaggle.com/code/kaitofukami/159-160-vs-frontier-v20-weed-slip-recovery) | projected shed, identity-free safety, bounded WEED recovery, exact artifact contract |
| [Rayk Kretzschmar — Findings from Zero to Top Meta](https://www.kaggle.com/code/raykkretzschmar/kaggriculture-findings-from-zero-to-top-meta) | complete-route memory, both-seat promotion, and honest replay/LB boundaries |
| [Rayk Kretzschmar — Rank Your Agent](https://www.kaggle.com/code/raykkretzschmar/kaggriculture-rank-your-agent) | current-engine and both-seat evaluation discipline |
| [prvsiyan — The Moon Counts Melons](https://www.kaggle.com/code/prvsiyan/kaggriculture-frontier-the-moon-counts-melons) | market timing as the differentiator inside a converged production basin |
| [prvsiyan — Kaggriculture Baseline / Soil](https://www.kaggle.com/code/prvsiyan/kaggriculture-baseline) | preserve a coherent full schedule; avoid isolated edits that break downstream labor |
| [beicicc — C20 Exact Replication Control](https://www.kaggle.com/code/beicicc/kaggriculture-c20-exact-replication-control) | exact-copy control and legal terminal inventory discipline |
| [flexonafft — Adaptive Replay Agent](https://www.kaggle.com/code/flexonafft/kaggriculture-adaptive-replay-agent) | field/market channel separation and actor-local recovery |
| [Georgy Mamarin — Daily Live Meta](https://www.kaggle.com/code/georgymamarin/kaggriculture-daily-replays-the-live-meta-report) | replay memory is time-stamped and must be refreshed |
| [cjlcjlcjl — What the Top Farms Do](https://www.kaggle.com/code/cjlcjlcjl/kaggriculture-what-the-top-farms-do-a-live-meta) | current Top-farm composition as a population diagnostic |
| [Muneeb — Cow Ranch with Fertilizer Probe](https://www.kaggle.com/code/muneeb2405/kaggriculture-cow-ranch-with-fertilizer-probe) | exact submission/seat trajectory provenance and engine-mismatch caution |

The deployed field/labor route is a real public replay trajectory from
**{base['team']}**, episode **{base['episode']}**, source seat **{base['seat']}**,
selected as the fit-only medoid of {base['fit_trajectories']} trajectories. Its
action hash is `{base['hash']}`. This is credited public replay provenance, not
claimed as a newly invented production plan.

No linked Notebook's compressed artifact is copied. Mechanisms were independently
reimplemented and ablated around the selected public route. Downstream forks
should preserve both the route provenance and these mechanism links.


## 4. Runtime model: nearest public route, then a set intersection

Each of the 30 fit-only medoids contributes two identity-free sequences:

1. a public farm signature at every step;
2. the set of products that route sold at that step.

The signature contains only public consequences of play:

- farm-hand count and actor positions;
- unlocked quadrant bitmask;
- crop, animal, pasture, coop, and low-weight WEED counts;
- public yield-unit totals.

Distance weights labor/unlocks most heavily, production counts next, actor
positions lightly, and stochastic WEED count at only `0.25`. At runtime:

```python
memory = nearest(public_opponent_signature, 30_fit_only_prototypes)

if memory.distance <= 48:
    predicted_collisions = memory.sell_set[current_step]
    collided = existing_route_sells & predicted_collisions
    market = collided_first_preserving_order(market)
```

The model is a one-nearest-neighbor policy memory, not a username lookup. The
30 runtime records are stripped to `{index, signatures, sales}` before
embedding. It cannot see the opponent's private shed or current simultaneous
order.


## 5. The safety invariant: change order, never inventory

The first conditional-memory prototype predicted a route's future SELL and
created a new early order. Identification worked, but intervention frequency
rose to roughly 28 preemptions per game and performance collapsed. This is the
same lesson as the earlier neural SELL-uplift experiment: predicting an action
does not establish the closed-loop value of taking it early.

The promoted controller enforces:

```text
ordinary turn:
  same farmer action
  same hand actions
  same BUY/HIRE orders
  same SELL products
  same SELL quantities
  only the order of existing market blocks may change

step 718:
  legally project DROP/PLACE into shed
  sell all remaining products in collision/value order
```

That constraint is visible in the development ablation:

| Complete option on current Konstantin route | Wins |
|---|---:|
| route unchanged | 51/53 |
| global exposure order | 52/53 |
| global two-step quarter preemption | 49/53 |
| conditional 5-neighbor order | 50/53 |
| **conditional nearest-route order** | **53/53** |

The selected memory made 682
actual order changes across 53 games. It abstained on
301 of
12614 SELL-containing queries
because no public prototype was close enough.


## 6. Frozen results and the old-public control

| Evaluation | v21.1 conditional memory | old public v21 | Paired old loss→new win | Paired old win→new loss |
|---|---:|---:|---:|---:|
| reserved option window | **53/53** | — | — | — |
| balanced unopened outer | **59/60** | 42/60 | 17 | 0 |
| full unopened outer | **177/180** | 123/180 | 55 | 1 |
| post-cutoff current population | **46/51** | 24/51 | 23 | 1 |

Full-outer mean margin improved from **3,560**
to **13,049**. On the post-cutoff set it
improved from **-841** to
**7,336**.

The 46/51 future result is lower than 177/180, which is exactly the drift signal
we should expect from a fast public meta. It is still a substantial paired
improvement over the artifact currently exposed on this page. This future set
was not used to change the frozen policy.


## 7. Failure audit: where the edge is already thinning

The unopened full outer lost only three rows, all against the Seb family:

```text
episode 90300722 | seat 1 | Seb (allegedly) | margin -15391
episode 90295581 | seat 1 | Seb (allegedly) | margin -13621
episode 90289799 | seat 1 | Seb (allegedly) | margin -2380
```

The later post-cutoff collection lost five:

```text
episode 90337286 | seat 0 | Ben Hamilton | margin -245
episode 90339454 | seat 0 | Seb (allegedly) | margin -5823
episode 90337991 | seat 1 | Seb (allegedly) | margin -4117
episode 90340128 | seat 1 | OceanMix | margin -1838
episode 90339385 | seat 0 | OceanMix | margin -4061
```

Ben's `-245` is a timing-scale miss; the repeated Seb and OceanMix losses are
more consistent with a stronger production/market route family than with one
bad terminal order. They are intentionally reported rather than patched after
seeing the holdout. The next model may use them as development data only if it
collects a newer untouched outer window.


## 8. What this does—and does not—solve

What improved:

- current route refresh repairs the dominant stale-route failure;
- both seats are evaluated separately;
- 30 routes are used at runtime without identity branches;
- matching is state-dependent, so non-identical route families receive
  different order choices;
- no ordinary SELL is invented and no ordinary quantity is changed;
- current engine `kaggle-environments==1.32.4` is used, including locked-tile
  movement and shed-capacity semantics.

What remains impossible or unproven:

- an exact deterministic copy receives the same logic; permanent domination
  of the exact same file is impossible without exogenous asymmetry;
- fixed replay opponents do not react to our counterfactual market changes;
- a nearest stored route can be wrong for a genuinely new policy;
- the field/labor channel is still a complete route, not a learned closed-loop
  optimizer;
- post-cutoff performance already shows faster decay than the original outer.

The next defensible step is a portfolio of complete field+market policies with
an online public-state value selector, trained on these now-development losses
and promoted only on newer submissions and both seats.


## 9. Exact public artifact

The submission artifact is **428,598 bytes**, Python-standard-library
only, SHA-256:

`d9dc24ce5429ec628ead0621a160bee90725350683d7dfcc4686fcaf511f3aab`

It embeds one 719-action current route and 30 identity-free public-state/sale
memories. The final cell reconstructs those exact bytes, verifies byte count
and hash, compiles the file, creates a root-level `submission.tar.gz`, and runs
exact self-play smoke tests. Self-play must tie; that is the symmetry sanity
check, not a leaderboard claim.

The competition submission is created from this Notebook version's `main.py`
output, preserving Notebook linkage.
