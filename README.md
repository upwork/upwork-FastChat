This is an internal Upwork fork of [FastChat](https://github.com/lm-sys/FastChat).

## Setup

Create a virtual env.

```bash
# Run this once after cloning.
python3 -m venv .venv
# Run this to activate the venv.
. .venv/bin/activate
# Run this to deactive the venv.
deactivate
```

Install

```bash
pip install -e ".[model_worker,webui]"
```

Run web UI

```bash
python3 -m fastchat.serve.gradio_web_server --controller-url= --model-list-mode=reload --register-api-endpoint-file=fastchat_api_endpoints.json
```

Note that to use OpenAI/Fireworks models the env vars `OPENAI_API_KEY` and `FIREWORKS_API_KEY` must be set.

