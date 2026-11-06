# Name That LLM: free-proxy GPT and Gemini generations (lucasboesen)

These are single-paragraph passages from the 8 GPT and Gemini labels of the
[Name That LLM](https://www.kaggle.com/competitions/name-that-llm) competition. They were generated through the
**Kaggle Benchmarks Model Proxy on Kaggle's free daily AI quota**, which every participant has; no paid API key was used.
No competition text, id or prompt was sent to any model, and none is included here.

This data is used by our public notebook [LoRA x Adaptive Centering](https://www.kaggle.com/code/lucasboesen/lora-x-adaptive-centering),
where it trains a GPT-family gate.

## Protocol
- **Prompts:** the CC BY 4.0 prompt bank of
  [seyitkaangunes/ntl-extra-api-passages](https://www.kaggle.com/datasets/seyitkaangunes/ntl-extra-api-passages)
  (`prompt_id` g0000-g0999). This version covers g0000-g0059, g0400-g0729 plus 1482 prompt-bank-v2 prompts (1872 prompts, 7,906 passages).
  - Every prompt here has all 8 labels. Together with Seyit's 5 Claude passages, each prompt has 13 of the 18 labels.
  - We use the same prompts as abdullah1afify/ntl-generations (g0000 upward) and dophuchung/ntl-proxy-generations (g0999 downward). Our batches from g0400 on stay clear of both.
- **Request:** one user message, no system prompt: `<prompt>` + a blank line + `Write a single paragraph of about N words, with no title or headings.`
  - One fresh conversation per call.
  - Reasoning effort `low`.
  - `max_tokens` 4096.
  - No temperature, top_p or seed.
- **Length:** `N` (`target_words`) is drawn per (prompt, label) from the word counts of the competition train passages (numbers only), with a fixed seed. The median ratio of produced to requested words is 1.00.
- **Models (proxy slug -> label):**
  - openai/gpt-5.5-2026-04-23 -> gpt-5.5
  - openai/gpt-5.6-luna, gpt-5.6-sol, gpt-5.6-terra
  - openai/gpt-6-astra
  - google/gemini-3.1-pro-preview, gemini-3.6-flash, gemini-3.8-flash

  The slug each run actually used is in `model_slug`.
- **Cleaning:** surrounding whitespace stripped. Nothing else.

## Columns (`gen_passages.csv`)
| column | meaning |
|---|---|
| `prompt_id` | id in the seyit prompt bank (the prompt text is in that dataset's `prompts.csv`) |
| `model` | competition label |
| `model_slug` | proxy model identifier of the run |
| `target_words` | the requested N |
| `n_words` | whitespace word count of the passage |
| `setting` | reasoning effort sent (`low`) |
| `batch` | generation batch |
| `text` | the passage |

## Things to know
- **Typography shift.**
  - Train Gemini passages contain no em-dash. Here, 13-18% of the Gemini passages contain one (the other public proxy datasets show the same).
  - The GPT passages use curly quotes a little more often than train (60-67% vs 47-58%).
  - Rules that rely on train-only punctuation invariants may therefore not hold on proxy text, or possibly on the hidden test.
- **What it was useful for (our CV, prompt-grouped folds).** Extra prompts must carry (almost) all reachable labels.
  - Adding Claude-only prompts to an 18-way head lowered Claude recall: the topic becomes a label shortcut.
  - Adding our full 13-label prompts helped a frozen-DeBERTa head by +0.010.
  - A 5-way GPT gate trained on train GPT rows plus these passages helped. The same gate without them did not.

## License
CC BY 4.0. The prompts come from seyitkaangunes/ntl-extra-api-passages (CC BY 4.0). Please credit both.
