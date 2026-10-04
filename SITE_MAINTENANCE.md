# Site maintenance guide

This guide is for people changing the implementation, configuration, or visual behaviour of the site. For writing and publishing an ordinary post, see [`README.md`](README.md).

## Architecture at a glance

- `hugo.toml` contains the canonical site URL, PaperMod theme selection, global parameters, menu, taxonomy settings, and Goldmark/MathJax Markdown delimiters.
- `content/` holds pages and posts. Posts are under `content/posts/`; `_index.md` files create section pages.
- `layouts/` contains site-level Hugo template overrides. Hugo uses these in preference to the matching theme templates.
- `layouts/baseof.html` is the site shell. It loads `layouts/partials/head.html`, defines the `main` block, and loads the theme header/footer.
- `layouts/posts/list.html` renders post and topic listings. It selects posts from `site.Params.mainSections` (`["posts"]` in `hugo.toml`) and uses `post_card.html` for list entries.
- `layouts/_default/single.html` is the single-post template. It invokes `extend_body.html` for page-level content such as the AI disclosure.
- `layouts/partials/` contains site-specific partials. `extend_head.html` conditionally loads chemistry/math assets; `math.html` selects KaTeX or MathJax; `katex.html` contains KaTeX assets/configuration; `chem.html` renders SMILES structures.
- `assets/`, `static/`, and `images/` provide theme-processed assets, directly copied static files, and site image resources respectively. `themes/PaperMod/` is the theme dependency; avoid editing it directly when a project override will work.

Hugo templates, theme partials, and front matter are coupled: when moving or renaming a partial, update every call site and verify with a build.

## Local development and validation

Install Hugo Extended at version 0.146.0 or newer (the base template enforces this minimum). The repository declares its Go/Hugo modules in `go.mod` and theme dependency in `.gitmodules`.

```sh
hugo server -D
```

Build the production site with:

```sh
hugo --minify
```

Inspect the output in `public/` and resolve build errors before publishing. The repository has a `test.py`, but review it before relying on it as a complete test suite. Do not hand-edit generated `public/` output; change source content/templates/config and rebuild.

## Common changes

### Add or change a menu item

Edit `[[menu.main]]` entries in `hugo.toml`. `weight` determines order; use a site-relative URL such as `/posts/`.

### Change the home page or post lists

The home page uses the theme's profile mode parameters in `hugo.toml`. Post list selection is customized in `layouts/posts/list.html`. `mainSections` determines which content types are considered blog posts; keep it aligned with the content layout and listing logic.

### Enable maths or chemistry on a page

A page opts in with `math: true` or `smiles: true` in front matter. `mathEngine` may be `katex` or `mathjax`, and defaults to KaTeX. Keep `layouts/partials/math.html` responsible for engine selection and its engine-specific setup; `extend_head.html` should remain the page-level opt-in/dispatch. KaTeX and MathJax are loaded from public CDNs, so a network connection is required in the visitor's browser. `hugo.toml` configures Goldmark passthrough delimiters; keep those consistent with the client-side renderer.

The chemistry partial loads SmilesDrawer from unpkg and looks for `[data-smiles]` elements. It observes PaperMod's `data-theme` attribute so drawings can be refreshed for light/dark mode.

### Styling and static resources

- Add theme-compatible CSS overrides under `assets/css/extended/` so Hugo bundles them with the theme styles.
- `static/` files are copied verbatim to the site root. The local math stylesheet is `static/css/math.css` and is linked from `layouts/partials/head.html`.
- Prefer a project-level override or asset over changing files under `themes/PaperMod/`; theme updates can overwrite direct theme edits.

## Editing partials safely

- `head.html` is a site override of the theme's head partial. It includes metadata, styles, theme behavior, and the `extend_head.html` hook. If changing this large template, compare against the theme version so upstream security/accessibility fixes aren't accidentally lost.
- Keep page-specific scripts conditional so they do not load on every page.
- Use Hugo template comments (`{{/* ... */}}`) for explanations that should not appear in rendered HTML.
- Partials that depend on `.Params` expect a Hugo page context. If calling one with a dictionary, update the partial to read the dictionary explicitly.
- Avoid duplicating the same CSS/JS include in multiple partials; identify one owner and keep its call path clear.

## Publishing and deployment

The canonical URL is configured as `baseURL` in `hugo.toml`. Check that it remains correct before a production build. GitHub Pages deployment is managed by the repository's workflows under `.github/`; inspect those workflow files if changing build or deployment behavior. Do not commit generated output or alter deployment configuration without confirming the existing workflow's expectations.
