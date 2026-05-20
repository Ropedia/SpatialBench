# SpatialBench Project Page

This repository contains a static, GitHub Pages-ready frontend for the SpatialBench paper and benchmark results.

## What is included

- `index.html` for the project page structure.
- `styles.css` for the responsive visual design.
- `script.js` for leaderboard rendering, filtering, sorting, and figure/dataset sections.
- `data/results.js` for the extracted main experimental results and dataset summary.
- `assets/` for paper figures copied from the LaTeX source.

## Local preview

Open `index.html` directly in a browser, or serve the directory with any static file server:

```bash
python3 -m http.server 8080
```

## Deploy on GitHub Pages

1. Push this directory to a GitHub repository.
2. In the repository settings, enable GitHub Pages.
3. Select the branch and root folder that contains `index.html`.

## Updating results

The leaderboard data lives in `data/results.js`. Replace the entries there when the LaTeX tables are updated.

## License

The frontend code is released under the MIT License. Dataset and model assets remain subject to their original licenses.
