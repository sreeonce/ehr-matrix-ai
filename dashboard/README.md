# EHR-Matrix AI — dashboard

A Next.js dashboard showing predicted sepsis risk over time for a handful of
example ICU patients, built from the fusion model trained earlier this week.
There is no backend: all predictions are pre-computed into
`data/patients.json` by `export_dashboard_data.py`, and the app just reads
that file.

## 1. Generate the data (run once, or whenever the model/data changes)

From your `ehr-project` folder (the one with `modeling_table.pkl`,
`notes.pkl`, `note_embeddings.npy`, `model_fusion.pt`), with this
`dashboard/` folder placed inside it:

```
python export_dashboard_data.py
```

This picks 5 example patients (4 who deteriorate into sepsis, 1 who stays
stable) and writes `dashboard/data/patients.json`.

## 2. Run the dashboard locally

```
cd dashboard
npm install
npm run dev
```

Open http://localhost:3000.

## 3. Deploy (for Friday 10 Oct)

The simplest free option for a Next.js app:

1. Push this `dashboard/` folder to a GitHub repo.
2. Go to https://vercel.com, sign in with GitHub, click **New Project**,
   and import the repo.
3. Leave all settings default (Vercel auto-detects Next.js) and click
   **Deploy**.
4. You'll get a live `https://<project>.vercel.app` URL to share.

No environment variables or backend setup needed — `data/patients.json` is
bundled into the build.

## Notes

- Re-run `export_dashboard_data.py` any time you retrain the model or want
  different example patients, then redeploy (or just `npm run dev` again
  locally).
- The dashboard is read-only demo data — the "Edit" on physician notes only
  changes what's shown in your browser tab; it isn't saved anywhere.
