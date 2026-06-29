---
applyTo: "docs/**"
---

# Documentation conventions (docs\)

When writing or editing any file under `docs\`:

- **Always reference files and folders by their full path from the project root.**
  Example: write `src\localai\capabilities\image\text_to_image\adapter.py`, not
  `adapter.py`, not `text_to_image\adapter.py`, and not the Python dotted form
  (`localai.capabilities...`). A reader must be able to copy the path and find the
  file without guessing where it lives.
- Use **Windows-style backslash** separators in these root-relative paths.
- For installed third-party libraries, give a locatable path too — they install
  under `.venv\Lib\site-packages\<package>\` (e.g.
  `.venv\Lib\site-packages\diffusers\`). Don't list a library by bare name when
  the reader may need to find its files.
- For downloaded models / external data, give the real on-disk location (e.g. the
  Hugging Face cache under `%USERPROFILE%\.cache\huggingface\hub\`).
- **Exceptions — keep these as written:**
  - Genuine Python identifiers: the entry-point literal `localai.core.cli:main`
    and real `import` / `from ... import` statements inside code blocks.
  - Markdown link targets `[text](path)` must stay forward-slash (they are URLs).
  - ASCII tree diagrams may keep forward-slash directory indicators.
- If you add or change a Mermaid diagram, validate it renders before committing
  (backslash paths in labels are fine).
