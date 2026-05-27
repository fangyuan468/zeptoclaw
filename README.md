<p align="center">
  <img src="assets/mascot-no-bg.png" width="200" alt="Zippy — ZeptoClaw mascot">
</p>
<h1 align="center">ZeptoClaw — production fork</h1>
<p align="center">
  <strong>Production fork of <a href="https://github.com/qhkm/zeptoclaw">qhkm/zeptoclaw</a> with extra engineering for sandboxed deployment, ACP / AG-UI streaming, HITL approval, prompt-cache stability and a refactored agent state machine.</strong>
</p>
<p align="center">
  <a href="https://github.com/767829413/zeptoclaw"><img src="https://img.shields.io/badge/fork-767829413/zeptoclaw-3b82f6?style=for-the-badge" alt="Fork repo"></a>
  <a href="https://github.com/qhkm/zeptoclaw"><img src="https://img.shields.io/badge/upstream-qhkm/zeptoclaw-3b82f6?style=for-the-badge" alt="Upstream"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache%202.0-blue?style=for-the-badge" alt="License"></a>
</p>

---

## Repository layout

- `main` — tracks `upstream/main` (read-only mirror, never developed against directly).
- `production` — what we actually run. Branches off `main`, carries every feature listed below.
- `feat/*` — feature branches, opened against `production` via PR, deleted after merge.

The integration layer lives in a separate ops repository (`767829413/openshell-zeptoclaw-ops`) which orchestrates the OpenShell sandbox, deployment scripts, and global plans. **This repository contains only the ZeptoClaw agent code.**

## What this fork adds on top of upstream

61 commits / 8 themed deltas vs. `upstream/main`. Everything here is shipping on `production` today.

### 1. OpenShell sandbox integration

ZeptoClaw upstream supports six sandbox runtimes natively. Our deployment runs it inside **OpenShell** (a separate fork at `767829413/openshell`) for stronger isolation, per-request network policies and proxy-aware DNS. The wiring lives in this repo as:

- Host-managed config injection, restart wrapper, searxng / web fetch policies.
- Skills mounted into the workspace (no in-image bundling).
- Hardened MCP session handling with secure credential injection.
- Proxy-aware DNS fallback for `web_fetch` and friends, fail-soft errors with classified retry.

Commits: `feat: add openshell deployment configuration` → `deploy(openshell): host-managed config, searxng policy, restart wrapper`.

### 2. ACP and AG-UI streaming

Used by our Tauri client and headless ACP consumers. Adds an end-to-end push channel for thinking events, tool-call lifecycle, file artifacts and rich A2UI payloads on top of the upstream message bus.

Highlights:

- ACP stdio: header-only read loop, streamed prompt chunks, keep-typing without closing prompts, structured agent-error surfacing.
- AG-UI custom events: thinking start/elapsed/end, tool-call started/finished with structured outcome, file artifact lifecycle, A2UI rows.
- A2UI prompt suffix externalized to `prompts/a2ui_v0_9.md`, gated on channel capability, hardened against shell-tool chart fallbacks.
- Output prioritization: A2UI first, mermaid fallback second, plain text last.

Done plans: `step-5-agui-endpoint.md`, `step-7-tauri-client-mvp.md`, `step-8-agui-generative-ui.md`, `tauri-file-editor.md`.

### 3. HITL approval broker and direction-scoped safety

Our Discord and gateway deployment need a deterministic approval flow with thread-level context.

- Approval broker resolves requests by id (prevents cross-talk under concurrent threads).
- HardFloor rules, slash commands, thread-level approval mode.
- Direction-scoped policy checks: `InputOnly` rules no longer block output content.
- Discord inbound dedupe and nonce idempotency on the broker side.

Done plan: `zeptoclaw-thread-approval.md`.

### 4. Prompt cache stability

Provider prompt-cache (DeepSeek / SiliconFlow / OpenAI / Anthropic) is a major cost lever. Upstream's `RuntimeContext::render()` re-emits `Current time / Timezone` every turn which invalidates the prefix hash. We cut that out and plumb cache breakdown through `Usage` / metrics / cost.

- `perf(prompt-cache)`: drop minute-level fields from runtime context.
- `feat(observability)`: prompt-cache hit / miss tokens surfaced via `Usage`, metrics and cost estimator.
- `fix(providers/openai)`: fall back to legacy `prompt_cache_hit_tokens` for non-conforming gateways.
- `feat(providers)`: SiliconFlow registered as OpenAI-compatible provider with the right cache headers.

### 5. Token cost optimization (P1–P4 landed)

A multi-stage program tracked under `docs/plans/doing/2026-05-21-token-cost-optimization.md`. Goal: cut input tokens 40–60 % on long sessions without changing behavior.

| Phase | What landed | Status |
|---|---|---|
| **P1 / P2** | Compress base + A2UI guidance, externalize prompt files | shipped |
| **P3** | Split runtime context into L2 / L4 layers for prefix-hash stability | shipped (PR #2) |
| **P4** | Lazy tool schema: ship a thin schema by default, expand on demand via `internal__get_tool_schema` | shipped (PR #1) |
| **P5** | Anchored rolling summary | **paused → next** |

Done plan: `lazy-tool-schema.md`.

### 6. Agent loop state machine refactor

Driven by a production bug (`empty reply from ACP prompt` — the loop was treating `has_tool_calls == false` as a successful turn). Completed in seven PRs over two weeks.

| PR | Phase | Outcome |
|---|---|---|
| #2 | Phase 1 | `TurnOutcome` classification, reject empty / provider-markup final answers |
| #3 | Phase 1 patch | Streaming markup guard — buffer `Delta` starting with `<` until `Done` classifies |
| #4 | Phase 2 | `FinalSynthesis` state — silent synthesis instead of empty reply |
| #5 | Phase 3 | `ToolObservation` / `ToolObservationKind` normalization |
| #6 | Phase 4 alt | Extract 8 themed helper modules from `loop.rs` |
| #7 | Phase 4.1 | `Harness<'a>` newtype + move `process_message` |
| #8 | Phase 4.2 | Move `process_message_streaming`, return `loop.rs` to a dispatch shell |

Result: `loop.rs` 6107 → 3533 lines (−2574), new `harness.rs` 2667 lines. Public API unchanged. `Harness<'a>` is now the natural mount point for the next phase (anchored summary / context state compression).

Done plan: `agent-loop-state-machine-refactor.md`.

### 7. BM25 memory backend

Workspace memory upstream was a plain key-value store. We added a BM25-scored retrieval backend and a `MEMORY.md` template for seed memory.

### 8. Web fetch hardening

Bounded retry with classified errors (DNS, TLS, HTTP status family, timeout). Proxy-aware. Fail-soft so a single bad URL doesn't blow up a tool loop.

---

## Project plans

Plans live in the ops repo (`docs/plans/` under `767829413/openshell-zeptoclaw-ops`). Status as of the latest sync:

### In progress (`doing/`)

| Plan | What |
|---|---|
| `2026-05-21-token-cost-optimization.md` | P5 anchored rolling summary — paused waiting on state machine, now unblocked |
| `step-9-thread-files.md` | Per-thread file workspace and artifact tracking |
| `workspace-fs-realtime.md` | Real-time workspace fs events to the Tauri client |

### Queued (`todo/`)

| Plan | What |
|---|---|
| `context-state-compression.md` | Chat / Task context modes with `TaskState` and `ConversationSummary` — follow-on to P5 |
| `provider-native-toolcall-fallback.md` | Fallback path when a provider drops native tool-call schemas |
| `step-4-management-api.md` | Management API for the Tauri console |
| `step-6-tauri-console.md` | Tauri console UI |
| `zeptoclaw-loop-modularization.md` | Next-pass `loop.rs` modularization (after state machine settles) |

### Recently closed (`done/`)

`agent-loop-state-machine-refactor.md`, `lazy-tool-schema.md`, `step-5-agui-endpoint.md`, `step-7-tauri-client-mvp.md`, `step-8-agui-generative-ui.md`, `tauri-file-editor.md`, `workspace-path-safety.md`, `zeptoclaw-thread-approval.md`, plus four `openshell-fork-*` plans driven from the openshell repo.

---

## Working with this fork

### Branches and PR flow

```bash
# Bring main up to date with upstream
git fetch upstream
git checkout main
git merge --ff-only upstream/main
git push origin main

# Start a feature from production
git checkout production
git pull origin production
git checkout -b feat/my-thing production

# After work + tests + clippy + doc-test
gh pr create --base production --head 767829413:feat/my-thing
```

PRs target `production`. `main` only receives upstream merges; never push features directly to `main`.

### Build and verify

```bash
cargo build --lib                       # debug
cargo build --lib --release             # release binary (used by the openshell wrapper)

cargo test --lib --quiet                # ~3719 tests, ~30s
cargo test --doc agent                  # agent doctest subset
cargo clippy --lib --all-targets        # must be zero new warnings vs. production

# Quick sanity for any agent-loop change
cargo test --lib agent::                # focused subtree
```

Baseline reference numbers on current `production`:

- `cargo test --lib --quiet`: **3719 passed, 0 failed, 5 ignored**
- `cargo test --doc agent`: **22 passed, 0 failed, 7 ignored**
- `cargo clippy --lib --all-targets`: 0 new warnings (6 pre-existing in `runtime/factory.rs`)

### Code map (delta vs. upstream)

The biggest structural change is in `src/agent/`:

```
src/agent/
├── loop.rs              # dispatch shell + entry points (start / stop / try_queue_or_process)
├── harness.rs           # state-machine runner: Harness<'a> { agent: &AgentLoop }
│                        #   - process_message            (non-streaming)
│                        #   - process_message_streaming  (streaming)
├── turn.rs              # TurnOutcome classification, StreamingMarkupGuard
├── observations.rs      # ToolObservation / ToolObservationKind
├── synthesis.rs         # FinalSynthesis state
├── format.rs            # presentation helpers
├── loop_events.rs       # tool-call event publishing, ThinkingScope
├── file_artifact.rs     # file artifact payload + custom event emission
├── tool_helpers.rs      # loop-guard, sequential exec, approval routing
├── inbound.rs           # inbound message -> Message conversion
└── context.rs           # layered prompt (L1/L2/L3/L4), runtime context render
```

`harness.rs` is where the next round of work (anchored summary, task state) will land — it owns one borrow of `AgentLoop` per turn and can absorb owned state without touching the public API.

---

## Upstream feature reference

Everything below is upstream behavior that `production` inherits unchanged. Full docs at <https://zeptoclaw.com/docs/>.

### Core

| Feature | What it does |
|---|---|
| **Multi-Provider LLM** | 18 providers (Anthropic, OpenAI, OpenRouter, Gemini, Vertex, Groq, DeepSeek, xAI, NVIDIA, Azure, Bedrock, Kimi, Zhipu, Qianfan, Novita, Liquid, Ollama, vLLM) with SSE streaming, retry with backoff + budget cap, auto-failover |
| **33 Tools + Plugins** | Shell, filesystem, grep, find, web, git, stripe, PDF, transcription, Android ADB and more |
| **Tool Composition** | Create tools from natural-language descriptions with `{{param}}` templates |
| **Agent Swarms** | Delegate to sub-agents with parallel fan-out, aggregation, cost-aware routing |
| **Library Facade** | `ZeptoAgent::builder().provider(p).tool(t).build()` for embedding |
| **Batch Mode** | Process prompts from text / JSONL files |
| **Agent Modes** | Observer / Assistant / Autonomous — category-based tool access |

### Channels and integration

Telegram, Slack, Discord, WhatsApp (Web + Cloud), Lark, Email, Webhook, Serial, ACP, plus plugin channels. Per-chat persona, hooks (`before_tool` / `after_tool` / `on_error`), cron and heartbeat, workspace + long-term memory.

### Security and ops

Six sandbox runtimes (Docker, Apple Container, Landlock, Firejail, Bubblewrap, native), prompt injection detection, secret leak scanner, policy engine, input validator, shell blocklist, SSRF prevention, chain alerting, tool approval gate, token budget + per-model cost, Prometheus / JSON metrics, structured logging, self-update, loop guard, multi-tier context trimming, session repair, config hot-reload, `HAND.toml` agent profiles, multi-tenant.

### Install (upstream)

```bash
curl -fsSL https://raw.githubusercontent.com/qhkm/zeptoclaw/main/install.sh | sh
brew install qhkm/tap/zeptoclaw
docker pull ghcr.io/qhkm/zeptoclaw:latest
cargo install zeptoclaw --git https://github.com/qhkm/zeptoclaw
```

**This fork is not packaged for end-user install.** We build release binaries from `production` and deploy them via the openshell wrapper. See `openshell-zeptoclaw-ops/openshell.sh`.

---

## Contributing to this fork

Open an issue or PR against `production`. Follow the patterns from the state-machine refactor (PR #2–#8) for any significant change:

1. Each phase ships as one PR with zero behavior change relative to its predecessor.
2. Cover with `cargo test --lib`, `cargo test --doc agent` and `cargo clippy --lib --all-targets`.
3. Provide a review checklist in the PR description (we did this for each of #6 / #7 / #8).
4. Plan documents in the ops repo are the source of truth — update them when you start a phase and again when you close one.

## License

Apache 2.0 — see [LICENSE](LICENSE). Inherited from upstream.

## Upstream credits

ZeptoClaw is built by [Aisar Labs](https://aisar.ai). This fork retains every notice from upstream and contributes back via PRs to `qhkm/zeptoclaw` when changes are general-purpose.
