# Demo video

A 15-second animated demo of JevControl. It shows the same agent decisions going to a large LLM and to a small decision model, then compares accuracy, latency and confidence. It has no screen recordings and no voiceover. **The numbers in it are illustrative and were not measured.**

| file | use |
|---|---|
| `jevcontrol-demo.mp4` | 1920×1080, 30 fps, H.264 — main cut |
| `jevcontrol-demo.webm` | same, VP9, smaller — for the web |
| `jevcontrol-demo-square.mp4` | 1080×1080 — social feeds |
| `jevcontrol-demo.gif` | 720 px preview used in the top-level README |
| `jevcontrol-demo-poster.jpg` | still from the results scene |

The animation is code, not footage. `jevcontrol-demo.html` draws every frame with `window.draw(t)`. Open it in a browser to watch it loop. To render it again:

```bash
npm i -D playwright        # or use an existing install
node docs/demo/render.mjs  # writes docs/demo/jevcontrol-demo.mp4 (needs ffmpeg)
```
