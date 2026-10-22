# Kaggle execution

**Local-model GPU integration is verified:** 67 tests and six structured outputs passed on a
private Kaggle run. See [actual GPU evidence](../docs/kaggle_gpu_run.json),
[recorded outputs](../demo/gpu-recorded-preview.html), and [reproduction](../docs/local_inference.md).
These use original notes, not reviewed official-source benchmark gold. The earlier CPU smoke
and embedding checks below remain separate historical evidence.

The owner authorized Kaggle CLI execution. CLI 2.2.4 is installed in `.venv-kaggle`; existing OAuth
was refreshed successfully, and a private CPU notebook completed with 45 tests passing and three demo cases. See
[verification evidence](../docs/verification.md) for the observed run result.

## Connect your account without sharing a token in chat

Existing valid OAuth credentials at `~/.kaggle/credentials.json` are detected automatically:
the helper reuses the cached unexpired access token in its child-process environment, without
printing it, copying a refresh token, or requiring a new token. Username is read from the same
profile. Expired credentials are refreshed by the official SDK. Account authorization has already been provided by the owner; this is not a new login.

Only if no usable authentication exists, the repeatable two-stage wizard captures your public
username slug and a secret API token:

```bash
python3 -m pip install -r requirements-kaggle.txt
bash scripts/setup_kaggle.sh
python3 scripts/kaggle_task.py doctor --check-auth
```

It opens Kaggle settings and writes only ignored `.env.kaggle` with mode 600. It does not create
a public notebook, transmit an OpenAI key, or write GitHub secrets. Username comes from your profile
URL; a token comes from [Kaggle API settings](https://www.kaggle.com/settings/api).

Existing exported `KAGGLE_API_TOKEN` and `KAGGLE_USERNAME` also work. Legacy credentials are supported
by Kaggle CLI; this helper directs CLI configuration into ignored project-local storage by default.
Do not commit `.env.kaggle`, `access_token`, `kaggle.json` or downloaded private outputs.

## Prepare, run and retrieve

```bash
python3 scripts/kaggle_task.py prepare
python3 scripts/kaggle_task.py push
python3 scripts/kaggle_task.py status
python3 scripts/kaggle_task.py output
```

If using OAuth instead of a token, run `kaggle auth login` yourself with your chosen configuration
directory. Credentials must be available to the same environment as this helper.

Prepare does not need a login. It creates an embedded-source notebook and metadata under
`artifacts/local/kaggle/kernel/`; provide `--owner your-slug` if no username is configured.
The ID contains a source hash, the notebook is private, CPU-only, and internet-enabled. Pushing
uploads and runs that notebook. No paid OpenAI calls are enabled by default.

The source payload is allowlisted and excludes `.env`, tokens, `.git`, local official documents,
vector indexes, budgets, and experiment artifacts. Only code and original draft/demo data are uploaded.
The embedded source has a SHA-256 check and safe-path validation before extraction.

Default output: core unit-test log, benchmark draft audit and three **actually executed** original-note
demo cases. Kaggle does not automatically provide an OpenAI key or a license to collect source pages.

For the optional paid connectivity probe, add `OPENAI_API_KEY` in the notebook's Add-ons → Secrets,
grant this notebook access, then enable the clearly marked cell. The CLI cannot create Kaggle Secrets.
Do not represent this one-question probe as the held-out quality study. Follow the main evaluation
workflow for reviewed official sources and gold before measuring quality.

Kaggle is a batch notebook execution environment here, not a public host for the local HTTP app.
Open the local UI separately; inspect saved reports/outputs in the notebook or download them.

Primary references: [authentication](https://github.com/Kaggle/kaggle-cli/blob/main/skills/references/auth.md),
[kernel commands and Secrets](https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels.md),
[metadata](https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels_metadata.md).
