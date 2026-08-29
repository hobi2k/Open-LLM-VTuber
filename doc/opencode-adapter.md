# OpenCode Adapter

This fork can use a running OpenCode headless server as the LLM backend. The adapter creates an isolated OpenCode session for each turn, forwards the Open-LLM-VTuber conversation history, and streams only assistant text events back to the speech pipeline.

## 1. Start OpenCode

Run this from the Open-LLM-VTuber directory so the bundled `vtuber` agent is discovered:

```bash
opencode serve --hostname 127.0.0.1 --port 4096
```

OpenCode keeps ownership of provider credentials. Configure the provider and model in OpenCode itself; no provider API key is copied into Open-LLM-VTuber.

## 2. Configure Open-LLM-VTuber

In `conf.yaml`, select the adapter:

```yaml
agent_config:
  agent_settings:
    basic_memory_agent:
      llm_provider: 'opencode_llm'
      use_mcpp: False

  llm_configs:
    opencode_llm:
      base_url: 'http://127.0.0.1:4096'
      provider_id: 'omlx'
      model: 'Qwen3.8-27B-oQ4e-mtp'
      agent: 'vtuber'
      workspace_directory: '.'
      timeout: 300
      keep_sessions: False
      allow_tools: False
      server_username: null
      server_password: null
      interrupt_method: 'user'
```

`provider_id` and `model` must exactly match IDs returned by the OpenCode provider configuration. The defaults above match this installation's local oMLX model.

## Safety And Sessions

- `allow_tools: False` adds a deny-all permission rule to every generated OpenCode session. This is the recommended VTuber setting.
- `keep_sessions: False` deletes transient sessions after the answer completes, preventing the OpenCode session list from filling with one-shot requests.
- Set `keep_sessions: True` temporarily when debugging prompts or model responses.
- If the OpenCode server uses `OPENCODE_SERVER_PASSWORD`, set `server_password` and optionally `server_username` (default OpenCode username: `opencode`).

## Verification

Check the OpenCode server before starting the VTuber:

```bash
curl http://127.0.0.1:4096/global/health
```

Then start Open-LLM-VTuber normally:

```bash
uv run run_server.py
```
