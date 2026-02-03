# pi-agentic-compaction

A [pi](https://github.com/badlogic/pi-mono) extension that provides conversation compaction using a virtual filesystem approach.

Based on [laulauland's file-based-compaction](https://github.com/laulauland/dotfiles/tree/main/shared/.pi/agent/extensions/file-based-compaction), with improvements from [w-winter's fork](https://github.com/w-winter/dot314/tree/main/extensions/agentic-compaction) but removing repo prompt components i don't use. 

## Improvements over the original

- **External config.json** - configure models, thinking levels, and performance parameters without editing code
- **Thinking level support** - configurable per-model (`off`, `minimal`, `low`, `medium`, `high`, `xhigh`)
- **Concurrent tool execution** - faster compaction via `toolCallConcurrency`
- **ctx.modelRegistry** - supports extension-registered providers (not just built-in models)
- **Prompt injection safeguards** - summarizer treats `/conversation.json` as untrusted input
- **Portable grep syntax** - uses `grep -E` with `|` (zsh-compatible)
- **Slash command filtering** - finds actual first user request, ignores `/compact` etc.
- **Temp artifact filtering** - excludes `__tmp*`, `.tmp` files from "Files Modified" list
- **User `/compact <note>`** - pass custom instructions to bias the summary

## Installation

**From git (once pushed):**

```bash
pi install git:github.com/Whamp/pi-agentic-compaction
```

**From npm (once published):**

```bash
pi install npm:pi-agentic-compaction
```

**Local development (symlink):**

```bash
# Add to settings.json manually since pi install doesn't support local paths
# In ~/.pi/agent/settings.json:
{
  "packages": [
    "/home/will/projects/pi-agentic-compaction"
  ]
}
```

Or symlink into the extensions directory:

```bash
ln -s ~/projects/pi-agentic-compaction ~/.pi/agent/extensions/agentic-compaction
cd ~/.pi/agent/extensions/agentic-compaction && npm install
```

The extension is auto-discovered on next pi start. Use `/reload` to hot-reload after changes.

## How it works

When pi triggers compaction (manually via `/compact` or automatically near context limits):

1. Converts the conversation to JSON and mounts it at `/conversation.json` in a virtual filesystem
2. Spawns a summarizer agent with sandboxed bash/jq tools to explore the conversation
3. The summarizer follows a structured exploration strategy:
   - Count messages and check the beginning (initial request)
   - Check the end (last 10-15 messages) for final state
   - Find all file modifications (write/edit tool calls with successful results)
   - Search for user feedback about bugs/issues
4. Returns the summary to pi

### Why this approach?

**pi's default compaction** serializes the entire conversation and sends it to an LLM in one pass. This works for shorter conversations, but for long sessions (50k+ tokens), you pay for all those input tokens and the model may struggle with "lost in the middle" effects.

**This extension's approach** mounts the conversation as a file and lets the summarizer explore it with jq/grep. Only the queried portions enter the summarizer's context, keeping costs low for long conversations.

Example queries the summarizer runs:

```bash
# How many messages?
jq 'length' /conversation.json

# What was the first user request (ignoring slash commands)?
jq -r '.[] | select(.role=="user") | .content[]? | select(.type=="text") | .text' /conversation.json | grep -Ev '^/' | head -n 1

# What happened at the end?
jq '.[-15:]' /conversation.json
```

**Trade-offs**:

- Cheaper for very long conversations (only loads what's queried)
- May miss context that a full-pass approach would catch
- Requires multiple LLM calls (one per tool use), but small/fast models make this quick

## Usage

This extension runs whenever pi compacts a session.

You can pass a note after `/compact` to bias the summarizer:

```
/compact Focus on the auth changes and ignore the test refactoring
```

## Configuration

Edit `config.json` next to `index.ts`:

```json
{
  "compactionModels": [
    { "provider": "cerebras", "id": "qwen-3-32b" },
    {
      "provider": "anthropic",
      "id": "claude-haiku-4-5",
      "thinkingLevel": "low"
    }
  ],
  "thinkingLevel": "off",
  "debugCompactions": false,
  "toolResultMaxChars": 50000,
  "toolCallPreviewChars": 60,
  "toolCallConcurrency": 6,
  "minSummaryChars": 100
}
```

| Parameter                          | Description                                                                      |
| ---------------------------------- | -------------------------------------------------------------------------------- |
| `compactionModels`                 | Models to try in order (first with API key wins)                                 |
| `compactionModels[].thinkingLevel` | Optional per-model thinking level override                                       |
| `thinkingLevel`                    | Default thinking level (`off`, `minimal`, `low`, `medium`, `high`, `xhigh`)      |
| `debugCompactions`                 | Save debug artifacts to `~/.pi/agent/extensions/agentic-compaction/compactions/` |
| `toolResultMaxChars`               | Truncate tool output to keep summarizer context small                            |
| `toolCallPreviewChars`             | Characters of command to show in UI notifications                                |
| `toolCallConcurrency`              | Max concurrent shell tool calls per turn                                         |
| `minSummaryChars`                  | Minimum accepted summary length (guards against empty summaries)                 |

## License

MIT
