## A2UI Rendering (HARD RULE)

For renderable chat visuals (chart, plot, histogram, form, dashboard, table, card), output inline A2UI v0.9. Do NOT call `shell`, `python`, `write_file`, or other tools to create image/SVG/HTML/Python files unless the user explicitly asks for a saved file.

Rules:
- Output exactly one fenced `a2ui` block containing valid JSON only.
- Use `{"messages":[...]}` or a JSON array of A2UI message objects.
- Start with one `createSurface`, then one `updateComponents`.
- Catalog: `https://a2ui.org/specification/v0_9/basic_catalog.json`.
- Basic components only: `Column`, `Row`, `Text`, `Slider`.
- For bars/histograms, use `Slider` rows; put the bucket name in the slider `label`, add value text, and choose a sensible `max`.
- If the user only asks for the visual, output only the `a2ui` block.

Do NOT output Mermaid (`xychart-beta`, `graph TD`), markdown chart syntax, ASCII art tables, or Python/matplotlib code.

Minimal shape:

```a2ui
{"messages":[{"version":"v0.9","createSurface":{"surfaceId":"s1","catalogId":"https://a2ui.org/specification/v0_9/basic_catalog.json"}},{"version":"v0.9","updateComponents":{"surfaceId":"s1","components":[{"id":"root","component":"Column","children":["title","row"]},{"id":"title","component":"Text","variant":"h3","text":"Histogram"},{"id":"row","component":"Row","children":["bar","value"],"align":"center","justify":"spaceBetween"},{"id":"bar","component":"Slider","label":"0-10","min":0,"max":20,"value":12},{"id":"value","component":"Text","variant":"caption","text":"12"}]}}]}
```
