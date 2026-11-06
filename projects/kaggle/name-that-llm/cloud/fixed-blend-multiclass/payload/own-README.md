# Name That LLM: own-prompt generations (pilot + v22)

749 single-paragraph passages written by the candidate models of the Kaggle competition "Name That LLM", generated
by us from prompts we wrote ourselves. Published under CC BY 4.0 so that every participant has the same free access
(the competition rules on external data).

## Provenance statement

No competition passage, id or prompt was sent to any model. The prompts are our own (files `prompts_pilot.json` and
`prompts_v22.json`). Their topics follow only the six domains named on the public competition Description page
(philosophy, physics, mathematics, biology, computer science, creative prose). Word targets: pilot = hand-picked integers
45-250; v22 = uniform random integers 40-250 (seed 20261022). Each request is one user message built from a template in
`templates.json`; no system prompt, max_tokens 8000, all other sampling settings left at provider defaults.

## Batches and call paths

| batch | generated (UTC) | call path | models | rows |
|---|---|---|---|---|
| pilot_20261015 | 2026-10-15 | Kaggle Benchmarks Model Proxy | 13 (gpt-5.5, gpt-5.6-luna, gpt-5.6-sol, gpt-5.6-terra, gpt-6-astra, gemini-3.1-pro-preview, gemini-3.6-flash, gemini-3.7-flash, gemini-3.8-flash, claude-opus-4-6, claude-opus-4-8, claude-opus-5, claude-sonnet-5) | 242 |
| pilot_20261015 | 2026-10-15 | OpenRouter | 5 (claude-fable-5, qwen3.8-max, glm-5.3, glm-5.3-flash, kimi-k3) | 87 |
| v22_20261022 | 2026-10-22 | Kaggle Benchmarks Model Proxy | 12 (the pilot proxy list without gemini-3.7-flash) | 420 |

The exact model string sent with each request is in the `request_slug` column.

## Filtering

- Pilot: 360 passages were generated (18 models x 20 prompts). 31 were removed because they share at least one
  8-word sequence with a competition passage (topic coincidence on classic subjects; removed to keep the release free of
  competition text).
- v22: 432 passages were generated (12 models x 36 prompts). 12 were removed by our own quality filters: 1 truncated
  (finish_reason "length") and 11 by format and content-overlap checks.
- Every string in this release was checked to share no 8-word sequence with any competition passage.

## Caveat on gemini-3.7-flash

The pilot rows labelled gemini-3.7-flash were requested with the model string `gemini-3.7-flash` through the Kaggle
proxy on 2026-10-15. That string was not among the proxy's listed models; the response reported the same model name,
but we cannot verify which model version actually served these requests. The v22 batch does not include this model.

## Files

- `passages.csv`: id, batch, prompt_id, model (label = the requested model), text, target_words, template_id, route,
  request_slug, generated_on
- `prompts_pilot.json`, `prompts_v22.json`: our prompts (pid, words, task; v22 also domain and template index)
- `templates.json`: the three request templates (the pilot used template 0)

## Rows per model

| model | pilot_20261015 | v22_20261022 |
|---|---|---|
| claude-fable-5 | 19 | 0 |
| claude-opus-4-6 | 19 | 36 |
| claude-opus-4-8 | 19 | 36 |
| claude-opus-5 | 18 | 36 |
| claude-sonnet-5 | 18 | 34 |
| gemini-3.1-pro-preview | 20 | 34 |
| gemini-3.6-flash | 19 | 32 |
| gemini-3.7-flash | 20 | 0 |
| gemini-3.8-flash | 20 | 34 |
| glm-5.3 | 16 | 0 |
| glm-5.3-flash | 17 | 0 |
| gpt-5.5 | 18 | 36 |
| gpt-5.6-luna | 18 | 34 |
| gpt-5.6-sol | 18 | 36 |
| gpt-5.6-terra | 17 | 36 |
| gpt-6-astra | 18 | 36 |
| kimi-k3 | 17 | 0 |
| qwen3.8-max | 18 | 0 |

## License and attribution

CC BY 4.0. Suggested attribution: "Name That LLM own-prompt generations (pilot + v22), Kaggle user earthmaker, October 2026".
Model outputs may also be subject to the terms of the respective model providers.
