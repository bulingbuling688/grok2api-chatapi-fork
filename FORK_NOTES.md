# Fork notes

This is a sanitized public fork snapshot of a Grok2API deployment used behind a New API gateway.

Local changes in this fork include:

- Console.x.ai SSO model support
- Grok 4.20 multi-agent model aliases
- SSO token pool selection and cooldown handling
- retry behavior for token-scoped 429 failures
- tests for console retry and SSO token extraction behavior

## Not included

The following runtime files are intentionally excluded and must never be committed:

- `data/token.json`
- `data/setting.toml`
- `data/temp/`
- `logs/`
- generated images/videos
- local backup files such as `*.bak-*`

Create runtime data locally through the application UI or your deployment automation.
