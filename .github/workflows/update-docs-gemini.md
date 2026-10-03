---
description: |
  Gemini-engine twin of update-docs.md (identical prompt). Use it when
  Copilot is unavailable or rate-limited: run it directly, or from the
  docs-automation hub with workflow=update-docs-gemini.lock.yml.
  Authenticated with the GEMINI_API_KEY repo secret (Gemini API free tier).

on:
  workflow_dispatch:

permissions:
  contents: read
  issues: read
  pull-requests: read

# Gemini CLI is pinned to 0.43.0: newer versions exit with
# "Invalid auth method selected" (code 41) behind the gh-aw API proxy.
# See https://github.com/github/gh-aw/issues/58445. Unpin once fixed.
#
# gemini-3.5-flash-lite is the only model whose free tier can complete a
# run: "auto" picks gemini-3.1-pro (free-tier limit 0) and gemini-3.5-flash
# allows only ~20 requests per quota window. With a paid key, use
# gemini-3.5-flash.
engine:
  id: gemini
  version: "0.43.0"
  model: gemini-3.5-flash-lite

network: defaults

tools:
  # No GitHub API tools: the agent reads the local checkout. This also keeps
  # each request smaller under the free tier's tokens-per-minute limit.
  github: false
  edit:
  # gh-aw writes allowlist entries as "git checkout:*" etc., which Gemini
  # CLI 0.43 does not match, so list the plain prefixes needed.
  bash: ["ls", "cat", "find", "grep", "head", "tail", "wc", "jq", "mkdir",
         "git ls-files", "git log", "git diff", "git status",
         "git branch", "git checkout", "git add", "git commit", "git config",
         "safeoutputs"]

safe-outputs:
  create-pull-request:
    title-prefix: "[docs] "
    labels: [documentation]
    draft: true
  # AI threat detection only runs on copilot, claude or codex engines, so it
  # is off here. Output is a docs-only draft PR reviewed before merge.
  threat-detection:
    engine: false

# Same safety caps as the Copilot workflow. Gemini's free tier pauses on its
# tokens-per-minute limit, so runs take longer: allow more time.
max-ai-credits: 20
max-turns: 60

timeout-minutes: 30
---

# Generate or Update Docs (Gemini)

You are the documentation maintainer for `${{ github.repository }}`. Your job is
to make the repository's documentation complete and accurate: create what is
missing, update what is outdated, and keep what is already correct.

## Work efficiently

This run has a small, fixed budget of model requests. When it runs out,
the run stops and nothing is published. So:

- **Do not use a todo or task-list tool.** Keep your plan in your head and
  start working.
- **Read only what you need:** the instructions file, the existing docs,
  dependency files and the main entry points. Do not open every file.
- **Write each file in one step.** For a new file or a large rewrite, create
  the whole file with a single call. Use small edits only for small changes.
- **Limit the scope of one run:** `README.md` plus at most two files in
  `docs/`. If more is needed, list it in the pull request for a later run.
- **Never repeat a call that just failed.** Change your approach or skip
  that file.

## 1. Understand the repository

- Read `.github/copilot-instructions.md` first, if it exists. It describes the
  project and what to ignore.
- List tracked files with `git ls-files`. Ignore virtual environments (`venv/`,
  `.venv/`), `node_modules/`, data files, model binaries, notebook checkpoints
  and generated output.
- Identify the language and stack, the entry points (scripts, apps, servers,
  CLIs), how dependencies are installed, how to run the project, how tests run,
  and any container or deployment setup (Dockerfile, compose, CI workflows).

## 2. Inventory the existing documentation

Find `README.md`, everything under `docs/`, and any other Markdown guides in
the repository root (for example `CONTRIBUTING.md` or `*_GUIDE.md`). For each
one, note what it covers and which statements are wrong, outdated or missing
compared with the code.

## 3. Create or update — per file

Apply this rule to every documentation file:

- **The file exists:** edit it in place. Keep content that is still accurate,
  including its structure, tone, links, badges and demo URLs. Correct anything
  the code contradicts, and add missing sections. Do not rewrite a file from
  scratch when targeted edits are enough.
- **Never duplicate a section.** Before adding a section, check whether the
  file already covers that topic (for example "Usage", "Running",
  "Troubleshooting"). If it does, edit that section in place instead of
  adding a second one.
- **If an edit fails, do not repeat it.** "No match found" means the text you
  are replacing is not in the file exactly as you wrote it (whitespace, line
  endings). Re-read the file and copy the text exactly, or write the whole file
  again. If the same file fails twice, leave it unchanged and mention it in
  the pull request.
- **The file does not exist:** create it.
- **Never delete** an existing documentation file. If one is obsolete, say so in
  the pull request instead.

### README.md (always handled)

Create it if missing, otherwise update it. It must contain:

1. Project name and a short description of what it does
2. Key features, based on the code
3. Tech stack
4. Prerequisites (language and runtime versions, tools)
5. Installation
6. Usage: how to run every entry point, with exact commands
7. Configuration: environment variables and config files, using
   `.env.example` or similar if present
8. Project structure: a short tree of the important folders and files
9. Testing, if tests exist
10. Docker or deployment, if present
11. Links to the files in `docs/`

### docs/ (create or update as needed)

If the `docs/` folder does not exist, run `mkdir -p docs` **before** creating
any file in it. The file-creation tool cannot create folders and fails with
"Parent directory does not exist".

- `docs/architecture.md`: components and modules, how data and control flow
  between them, and what each main module is responsible for.
- `docs/setup.md`: detailed local setup, dev container or Docker setup, and
  troubleshooting for common problems.

If the repository already has docs covering these topics under other names,
update those files instead of creating duplicates, and link to them from the
README.

## Rules

- Change documentation only: `README.md`, files under `docs/`, and existing
  Markdown guides. Never modify code, notebooks, dependencies, configuration
  or CI files.
- Base every statement on the code. Do not invent features, metrics, commands,
  environment variables or file names. Where something is unclear, write
  `TODO: confirm …` instead of guessing.
- Do not keep or copy claims from existing docs (feature lists, accuracy
  figures, "real-time" or similar wording) unless you find them in the code.
  Remove them, or mark them `TODO: confirm …`.
- Before calling a dependency optional, search for its imports. If any module
  or app imports it, it is required.
- Describe UIs from the code itself: list the actual tabs, pages or routes,
  and the files and models the app really loads. Do not assume an app uses
  an artifact just because another script produces it.
- When listing values defined in code (types, categories, cities, clusters),
  copy the full list or count from the code.
- In the pull request, claim that statements are verified only if you checked
  each one. List every `TODO` you left.
- Commit only the documentation files you created or changed. Never commit
  the pull request description, a summary or notes file (for example
  `docs/update-summary.md`): the description goes only into
  `create_pull_request`.
- Use plain, concise English and GitHub-flavoured Markdown. Use relative links.
- If all documentation is already accurate and complete, change nothing and
  do not open a pull request.

## Commit and open the pull request

The pull request is built from a local commit, so follow these steps in order
after all documentation edits are done:

1. Set a repository-local commit identity (run each as its own command):
   `git config user.name "github-actions[bot]"` and
   `git config user.email "github-actions[bot]@users.noreply.github.com"`.
   Do not change any other git configuration.
2. `git checkout -b docs/update-<short-topic>`
3. `git add` only the documentation files you created or changed.
4. `git commit -m "docs: <short summary>"`
5. Verify the commit exists with `git log --oneline -1`. It must show your
   commit message. If it does not, fix the problem and commit again. Never
   switch back to the default branch, and never use `--allow-empty`.
6. `git branch --show-current` and use exactly that name as `branch`.
7. Call `create_pull_request` once with `title`, `body` and `branch`. If it is
   not available as a direct tool, run it through the shell instead:
   `safeoutputs create_pull_request '<json>'`. To pass a long body, write it
   to `/tmp/gh-aw/agent/pr-body.md` (not into the repository) and build the
   JSON with `jq`, for example
   `jq -Rs --arg t "<title>" --arg b "<branch>" '{title: $t, branch: $b, body: .}' /tmp/gh-aw/agent/pr-body.md | safeoutputs create_pull_request .`
8. Stop. Do not push, and do not call `create_pull_request` again.

Use `noop` only if no documentation change was needed.

## Pull request content

Use a short descriptive title. In the body:

- list each file as **created** or **updated**, with a one-line summary;
- list every `TODO` you left;
- list any docs you think are obsolete but did not delete.
