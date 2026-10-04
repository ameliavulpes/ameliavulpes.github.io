# Hugo PaperMod

**A fast, clean, and responsive theme for [Hugo](https://gohugo.io/).**

[![hugo-papermod](https://img.shields.io/badge/Hugo--Themes-@PaperMod-blue)](https://themes.gohugo.io/themes/hugo-papermod/)
[![Minimum Hugo Version](https://img.shields.io/static/v1?label=Hugo&message=v0.146.0%2B&color=blue&logo=hugo)](https://github.com/gohugoio/hugo/releases/tag/v0.146.0)
[![Discord](https://img.shields.io/discord/971046860317921340?label=Discord&logo=discord)](https://discord.gg/ahpmTvhVmp)

> Based on [hugo-paper](https://github.com/nanxiaobei/hugo-paper/tree/4330c8b12aa48bfdecbcad6ad66145f679a430b3), with additional features and customization options.

<table>
	<tbody>
		<tr>
			<td>Live Demo</td>
			<td><a href="https://adityatelange.github.io/hugo-PaperMod/">adityatelange.github.io/hugo-PaperMod</a></td>
		</tr>
		<tr>
			<td>Documentation 📚</td>
			<td><a href="https://github.com/adityatelange/hugo-PaperMod/wiki">Github Wiki</a></td>
		</tr>
		<tr>
			<td>Example Site Source</td>
			<td><a href="https://github.com/adityatelange/hugo-PaperMod/tree/exampleSite">exampleSite branch</a></td>
		</tr>
		<tr>
			<td><a href="https://www.star-history.com/adityatelange/hugo-papermod"><img src="https://api.star-history.com/badge?repo=adityatelange/hugo-PaperMod&amp;theme=dark" alt="Star History Rank" /></a></td>
			<td><a href="https://ko-fi.com/H2H229ZWH"><img src="https://ko-fi.com/img/githubbutton_sm.svg" alt="ko-fi" /></a></td>
		</tr>
	</tbody>
</table>

<p align="center">
  <img src="https://user-images.githubusercontent.com/21258296/114303440-bfc0ae80-9aeb-11eb-8cfa-48a4bb385a6d.png" alt="Mockup image" title="Mockup"/>
</p>

---

## Features 💥

`☄️ Fast | ☁️ Fluent | 🌙 Smooth | 📱 Responsive`

- **Asset pipeline** -- Hugo's built-in asset generator with fingerprinting, bundling, and minification.
- **Three layout modes** -- [Regular](https://github.com/adityatelange/hugo-PaperMod/wiki/Features#regular-mode-default-mode), [Home-Info](https://github.com/adityatelange/hugo-PaperMod/wiki/Features#home-info-mode), and [Profile](https://github.com/adityatelange/hugo-PaperMod/wiki/Features#profile-mode).
- **Light and dark themes** -- Automatic switching based on browser preference, plus a manual toggle.
- **Multilingual support** -- Includes a built-in language selector.
- **Search** -- Client-side search powered by Fuse.js.
- **SEO optimized** -- Open Graph, Twitter Cards, and Schema.org structured data out of the box.
- **Cover images** -- Per-post cover images with responsive image support.
- **Table of contents** -- Auto-generated from heading structure.
- **Multiple authors** -- Native support for multi-author sites.
- **Social icons and share buttons** -- Configurable social links and per-post sharing.
- **Breadcrumb navigation**
- **Post archives and taxonomies**
- **Code block copy buttons** -- One-click copying with Chroma syntax highlighting.
- **Related post suggestions**
- **Zero JS build dependencies** -- No webpack, Node.js, or other tooling required.

| Topic                                                                                             | Description                                     |
| ------------------------------------------------------------------------------------------------- | ----------------------------------------------- |
| **[Installation guide](https://github.com/adityatelange/hugo-PaperMod/wiki/Installation)**        | Detailed installation and update instructions   |
| **[Features wiki page](https://github.com/adityatelange/hugo-PaperMod/wiki/Features)**            | In-depth explanations of all features           |
| **[FAQ wiki](https://github.com/adityatelange/hugo-PaperMod/wiki/FAQs)**                          | Common questions and configuration walkthroughs |
| **[Icons wiki](https://github.com/adityatelange/hugo-PaperMod/wiki/Icons)**                       | Documentation for social icons and share icons  |
| **[Variables wiki](https://github.com/adityatelange/hugo-PaperMod/wiki/Variables)**               | List of all available template variables        |
| **[Overiding templates](https://github.com/adityatelange/hugo-PaperMod/wiki/Template_Overrides)** | Guide to customizing templates without forking  |
| **[Releases](https://github.com/adityatelange/hugo-PaperMod/releases)**                           | Detailed history of releases                    |

---

## Performance ☄️

PaperMod consistently scores near-perfect results on [Pagespeed Insights](https://pagespeed.web.dev/report?url=https://adityatelange.github.io/hugo-PaperMod/).

<img width="481" height="116" alt="image" src="https://github.com/user-attachments/assets/497d831b-d143-4a46-bc11-b1d7f8ef4a83" />

---

## Support 🫶

- Star this repository to show your support.
- Share PaperMod with others who might find it useful.
- Sponsor the project on [GitHub Sponsors](https://github.com/sponsors/adityatelange) or [Ko-Fi](https://ko-fi.com/adityatelange).

---

## Special Thanks 🌟

- [Highlight.js](https://github.com/highlightjs/highlight.js)
- [Fuse.js](https://github.com/krisk/fuse)
- [Feather Icons](https://github.com/feathericons/feather)
- [Simple Icons](https://github.com/simple-icons/simple-icons)
- All contributors and supporters

---

# Writing for Amelia's blog

This site is built with Hugo and PaperMod. Blog posts live under `content/posts/`.

## Add a post

1. Create a Markdown file under `content/posts/`. Use a descriptive folder and file name, such as `content/posts/Machine Learning/New topic/index.md`. A page bundle (`index.md` in its own folder) is convenient when a post has images or other attachments.
2. Add front matter at the top of the file:

   ```yaml
   ---
   title: "A clear post title"
   date: 2026-10-04
   lastmod: 2026-10-04
   draft: true
   description: "A short description for previews and search."
   summary: "A short summary shown in the post list."
   tags: ["Example", "Notes"]
   categories: ["Machine Learning"]
   math: false
   ---
   ```

   Replace the examples with your post's details. Keep dates as `YYYY-MM-DD`. Set `draft: true` while writing, then change it to `false` or remove it to publish.

3. Write the article below the closing `---` in Markdown. Put page-specific images alongside `index.md` and refer to them with relative paths, e.g. `![Diagram](diagram.png)`.
4. Preview with `hugo server -D` (`-D` includes drafts), then commit and publish using the repository's normal GitHub workflow.

Optional front matter: `math: true` enables math rendering (`mathEngine: mathjax` selects MathJax; KaTeX is the default); `smiles: true` loads chemistry rendering for `[data-smiles]` elements; `AI_warning: true` displays the AI content notice; and `hideSummary: true` hides the post-list summary. For site implementation and configuration changes, see [`SITE_MAINTENANCE.md`](SITE_MAINTENANCE.md).
