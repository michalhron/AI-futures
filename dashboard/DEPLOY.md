# Deploying AI Vision Explorer (Streamlit)

## What the AI can vs cannot do for you

| Step | You (human) | Automated / local only |
|------|----------------|-------------------------|
| Create a GitHub account & new empty repo | Required | — |
| Paste `git remote` URL & run `git push` (browser or token login) | Required | I can run `git init` + `git commit` on your machine |
| Streamlit Cloud: connect GitHub & deploy | You click in the browser | — |
| Set `password` in Streamlit Secrets | You paste once | — |

There is no way for code running here to **publish to your GitHub** without **your** credentials (that would be a security problem). The commands below are everything else.

## Exact order (first time)

1. **On your Mac**, in the project folder:

   ```bash
   cd "/Users/michalhron/My Drive/ghent/AI visions/LLM_coder_tripple"
   git init
   git add .gitignore requirements.txt dashboard
   git commit -m "Add AI Vision Explorer dashboard"
   git branch -M main
   ```

2. **On github.com** → green **New repository** → name it (e.g. `ai-vision-explorer`) → **Create** (no README needed).

3. **Back in Terminal** (replace `YOUR_USER` and `YOUR_REPO`):

   ```bash
   git remote add origin https://github.com/YOUR_USER/YOUR_REPO.git
   git push -u origin main
   ```

   GitHub will ask you to sign in (browser or personal access token).

4. **Streamlit Cloud**: [share.streamlit.io](https://share.streamlit.io) → Sign in with GitHub → **New app** → select that repo → Main file path: `dashboard/app.py` → **Deploy**.

5. **App secrets** (Streamlit Cloud → your app → Settings → Secrets):

   ```toml
   password = "choose-a-shared-password"
   ```

6. **Data on the server**: Either upload `merged_analysis.csv` via Streamlit’s mechanisms, host the file somewhere and set **`DASHBOARD_CSV`** in app settings, or use Git LFS if the file is in the repo.

## What goes on GitHub

- Push the **code** (`dashboard/`, `requirements.txt`, etc.).
- Do **not** commit large CSVs or secrets. Add to `.gitignore`:
  - `old_with future types/data/processed/*.csv` (or your full data path)
  - `dashboard/.streamlit/secrets.toml`
  - `dashboard/article_publication.csv` (if it contains sensitive keys)

For the **large** `merged_analysis.csv`, prefer **not** committing it: use **Git LFS**, **private object storage** + `DASHBOARD_CSV`, or a hosted URL (see step 6 above).

## Streamlit Community Cloud (detail)

1. Sign in at [share.streamlit.io](https://share.streamlit.io) with GitHub.
2. **New app** → pick repo, branch, main file: `dashboard/app.py`.
3. Under **Secrets**, add at least:

   ```toml
   password = "your-shared-password"
   ```

4. Set **Advanced settings** if you need `DASHBOARD_CSV` pointing to a hosted URL or path.
5. Deploy. Share the `*.streamlit.app` URL with collaborators (password-protected).

## Publication lookup (`article_publication.csv`)

`merged_analysis.csv` has **no journal column**; outlet is inferred from **`full_text`**. To bake that into the repo (faster loads, stable labels), generate a lookup from the same data:

```bash
cd dashboard
python build_article_publication.py
```

This writes `article_publication.csv` (`__article_key`, `publication`). The dashboard **merges** live scan + this file; **CSV wins** on duplicate keys so you can hand-fix rows.

## Environment variables (optional)

| Variable | Meaning |
|----------|---------|
| `DASHBOARD_PASSWORD` | Password if not using `secrets.toml` |
| `DASHBOARD_CSV` | Absolute or relative path to `merged_analysis.csv` |

## Custom domain

Point a CNAME to Streamlit’s host (see Streamlit Cloud docs) or embed the app in your university site via **iframe** to the Streamlit URL.
