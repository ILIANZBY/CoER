# Anonymous review release

Checked on 2026-09-07 against the [ICLR 2027 Author Guidelines](https://iclr.cc/Conferences/2027/AuthorGuidelines) and [AI Policy for Authors](https://iclr.cc/Conferences/2027/AIPolicyForAuthors). See the [publication plan](publishing.md) for GitHub/Hugging Face account, model-metadata and reviewer-access gates. No remote publication is authorized by a successful local scan.

ICLR uses double-blind review. Supplementary code should be supplied as an anonymous archive or anonymous repository; code release is encouraged for reproducibility, but reviewers are not obliged to inspect it. Linked demos must preserve anonymity and must not track reviewers. The paper should explain reproducibility and point to the relevant method/code/data details. AI use must be disclosed in the submission and paper as required by the current policy; code assistance in this cleanup should be included in the authors' final disclosure.

## Prepare a source-only archive

```bash
python scripts/export_anonymous.py --output dist/corl-anonymous.zip
```

The exporter uses an explicit source allowlist, skips hidden files (except the reviewed root `.gitignore`) and runtime directories, rejects symlinks, omits PDFs/TeX/notebooks/model binaries and scans for common identifying paths, private addresses and credential patterns. The included `.gitignore` protects local artifacts and weights without hiding the `verl/models/` implementation. ZIP timestamps and permissions are normalized; no Git author/history metadata is copied. It refuses to overwrite an existing archive. A checksum manifest records the exported files.

Only the minimal training/evaluation source, necessary environment fixtures, original paper figures and licenses are included. General upstream documentation, old experiment outputs and local assistant configuration are not published. Legal notices stay intact. An automated scan is a safeguard, not proof of anonymity.

## Manual release gate

- Review the exported archive, not just the working tree. Search author names/handles, affiliations, private hostnames, contact addresses, project IDs, API keys and paths known to the authors.
- Inspect both paper figures and public-facing Markdown. Keep author lists, personal badges, tracking links and identifying citation metadata out of the anonymous release.
- Do not share the existing Git remote or history. If using a repository, create it from the reviewed export with anonymous account/commit metadata; check its hosting service for identity/analytics leakage.
- Keep source PDFs and old manuscript sources private unless separately reviewed. README figures remain usable without them.
- Release original datasets/checkpoints only after permission, safety and privacy review. Document missing assets honestly; do not claim complete reproduction while they are unavailable.
- Prepare a separate, reviewed result bundle. Strip private endpoints, mount paths and task traces; retain scientific provenance through anonymous aliases and hashes.
- Confirm third-party license requirements and keep required notices. Upstream attribution is not this submission's affiliation.
- Review every code change and run GPU/distributed smoke tests on the intended stack before submission. No training run or external benchmark has been launched by this cleanup.

The local source tree may still contain historical/local material excluded by the exporter. Removing a file from the working tree does not erase earlier commits. This procedure does not rewrite Git history or upload/publish anything.
