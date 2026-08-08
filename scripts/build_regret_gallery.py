"""Assemble the regret-mechanism figure gallery as a self-contained HTML page.

Embeds each PNG as a data URI (downscaled to a sensible web width) so the page
works with no external requests.  Figures that do not exist yet (Tier 3, still
on the cluster) render as labelled placeholders.
"""

import base64
import io
import json
import glob
from collections import defaultdict
from pathlib import Path

from PIL import Image

FIGDIR = Path("paper_v3/figures")
OUT = Path("paper_v3/regret_gallery.html")
MAX_W = 1700


def data_uri(path: Path):
    if not path.exists():
        return None
    im = Image.open(path).convert("RGB")
    if im.width > MAX_W:
        h = round(im.height * MAX_W / im.width)
        im = im.resize((MAX_W, h), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=92, optimize=True, subsampling=0)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/jpeg;base64,{b64}", im.width, im.height


def ring_facts():
    """Pull a few live numbers so the prose can't drift from the data."""
    facts = {}
    for tag, base in [("bast", "analysis/ring_regret_funwake"),
                      ("tp", "analysis/ring_regret_tp_funwake")]:
        series = defaultdict(dict)
        for fp in sorted(glob.glob(f"{base}/*/results.json")):
            d = json.load(open(fp))
            key = fp.split("/")[-2].rsplit("_n", 1)[0]
            for r in d["rings"]:
                series[key][r["n_farms"]] = r
        rows = series.get("a0.9_f1.0_d2", {})
        if rows:
            facts[tag] = {
                "rl_n1": rows[1]["regret_over_loss"],
                "rl_max": rows[max(rows)]["regret_over_loss"],
                "reg_n1": rows[1]["regret_pct"],
                "reg_peak": max(r["regret_pct"] for r in rows.values()),
                "n_peak": max(rows, key=lambda n: rows[n]["regret_pct"]),
                "reg_last": rows[max(rows)]["regret_pct"],
                "n_last": max(rows),
            }
    return facts


FIGS = [
    dict(file="ring_regret_fw.png",
         eyebrow="the observation",
         title="Regret does not rise with the number of neighbors",
         source="on disk",
         body="The starting point. Six series — three wind roses, two wake models, two buffer "
              "distances — sweeping the ring from one neighbor to eight. Regret climbs to a peak "
              "near four farms and then falls away, while the middle panel shows AEP loss doing "
              "something quite different. Filled markers are rings where every farm sits at the "
              "nominal gap; open markers are rings the packing constraint pushed outward."),
    dict(file="mech_decomposition.png",
         eyebrow="the identity",
         title="Regret is loss multiplied by what re-design can recover",
         source="on disk",
         body="The same data, split along the identity that governs it. The coloured band is the "
              "damage a neighbor-aware layout recovers — that is the regret. The grey band above it "
              "is damage no rearrangement can undo. Total loss keeps climbing to n=4 in every panel, "
              "but the recoverable share thins as it goes, and the product turns over."),
    dict(file="mech_phase_trajectory.png",
         eyebrow="the shape of it",
         title="The trajectory climbs a steep ray, then hooks onto a shallow one",
         source="on disk",
         body="Plotting regret against loss directly, with each point labelled by its ring size, "
              "turns the non-monotonicity into a visible curl. Dotted rays mark constant recoverable "
              "fraction. A system where regret simply tracked damage would run straight out along one "
              "ray; instead every path drifts to shallower rays as neighbors accumulate. The TurboPark "
              "panel shows it most clearly, because its regret signal is far above the multistart noise."),
    dict(file="mech_escape_rose.png",
         eyebrow="the geometry",
         title="Where the escape routes go",
         source="on disk",
         body="The bearings each neighboring farm actually subtends at the target, computed from the "
              "production ring geometry rather than sketched. Note the honest complication: at four "
              "farms only a fifth of the compass stays open yet recoverability is still 0.56, while at "
              "eight farms more bearings are open — because packing forced the ring out to 132D — and "
              "recoverability has fallen to 0.38. Open angle alone does not explain the effect; distance "
              "is entangled with it. Separating the two is what the pending runs are for."),
    dict(file="mech_two_paths.png",
         eyebrow="the unification",
         title="Three interventions, one governing quantity",
         source="on disk + gbar",
         body="Design regret is the value of knowing about damage in advance, and information is only "
              "worth something if you can act on it. The question is what closes off the acting. Three "
              "different interventions answer it together: concentrating the wind onto one axis leaves "
              "an escape route open, dividing a fixed neighbor capacity into smaller farms opens more, "
              "and piling on additional full-size farms closes them. The common thread is not symmetry — "
              "an earlier reading of this study said it was, and the third panel disproved it — but how "
              "much wake mass sits in the directions that carry energy, and how concentrated that mass is."),
    dict(file="mech_displacement.png",
         eyebrow="the mechanism, literally",
         title="Where re-design actually moves the turbines",
         source="gbar",
         body="Arrows from each turbine's liberal position to where the neighbor-aware optimizer puts "
              "it. The prediction was that escape moves should shrink as the ring closes, and they do: "
              "the median turbine travels 6.6D with one neighbor and 5.2D with eight, falling at every "
              "step, while the recoverable fraction falls 0.33 to 0.18 alongside it. One caveat worth "
              "stating plainly — the liberal and conservative layouts are independent multistart optima, "
              "so turbine index means nothing between them; the pairing here is the assignment that "
              "minimises total travel. Pairing by index instead reports a spurious 55D of movement that "
              "is mostly relabeling. Shown for Bastankhah, whose layout series completed first."),
    dict(file="mech_angular_spread.png",
         eyebrow="the controlled test",
         title="Angular spread with everything else held still",
         source="gbar",
         body="The test that overturned the tidy version of this story. Total neighbor capacity, total "
              "area and buffer gap are all held fixed while the capacity is divided into n smaller farms "
              "around the target — identical 50-turbine copies physically cannot ring closer than about "
              "44D past four of them, so scaling down was the only way to hold distance constant. "
              "Loss behaves as expected, rising under a bidirectional rose and falling under a "
              "unidirectional one as wake moves into or out of the directions carrying energy. But the "
              "recoverable fraction rises in every case, the opposite of the encirclement ring, because "
              "each individual corridor is now narrower and easier to slide out of. So what closes escape "
              "routes is concentrated wake mass, not angular symmetry."),
]


def main():
    facts = ring_facts()
    tp = facts.get("tp", {})
    cards = []
    n_built = 0
    for f in FIGS:
        res = data_uri(FIGDIR / f["file"])
        pending = res is None
        if not pending:
            n_built += 1
            uri, w, h = res
            media = (f'<img src="{uri}" width="{w}" height="{h}" alt="{f["title"]}" '
                     f'loading="lazy">')
        else:
            media = ('<div class="pending-slot">'
                     '<span class="pending-mark">awaiting cluster results</span>'
                     f'<code>{f["file"]}</code></div>')
        badge = ("on disk" if f["source"] == "on disk" else "new gbar run")
        badge_cls = "chip-disk" if f["source"] == "on disk" else "chip-gbar"
        state = "is-pending" if pending else ""
        cards.append(f"""
      <figure class="card {state}">
        <div class="card-head">
          <p class="eyebrow">{f['eyebrow']}</p>
          <h2>{f['title']}</h2>
          <span class="chip {badge_cls}">{badge}</span>
        </div>
        <div class="mount">{media}</div>
        <figcaption>{f['body']}</figcaption>
      </figure>""")

    n_pending = len(FIGS) - n_built
    rl1 = tp.get("rl_n1", float("nan"))
    rl8 = tp.get("rl_max", float("nan"))
    npk = tp.get("n_peak", 4)
    rpk = tp.get("reg_peak", float("nan"))
    rlast = tp.get("reg_last", float("nan"))
    nlast = tp.get("n_last", 8)

    html = f"""<title>Why design regret peaks and falls</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
  :root {{
    --ground:      #eef1f4;
    --surface:     #ffffff;
    --surface-alt: #e4e9ee;
    --ink:         #16202a;
    --ink-soft:    #3d4c5a;
    --muted:       #64748350;
    --muted-text:  #5c6b7a;
    --rule:        #cdd6de;
    --accent:      #b26a12;
    --accent-soft: #f0e2cd;
    --serif: "Iowan Old Style", "Charter", "Palatino Linotype", Palatino, "Book Antiqua", Georgia, serif;
    --mono:  ui-monospace, "SF Mono", SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace;
    --measure: 64ch;
    --wide: 1180px;
  }}
  @media (prefers-color-scheme: dark) {{
    :root:not([data-theme="light"]) {{
      --ground:      #0f151b;
      --surface:     #18212a;
      --surface-alt: #1e2933;
      --ink:         #e6ecf2;
      --ink-soft:    #c2ceda;
      --muted-text:  #8ea0b1;
      --rule:        #2b3843;
      --accent:      #e2a049;
      --accent-soft: #2e2519;
    }}
  }}
  :root[data-theme="dark"] {{
    --ground:      #0f151b;
    --surface:     #18212a;
    --surface-alt: #1e2933;
    --ink:         #e6ecf2;
    --ink-soft:    #c2ceda;
    --muted-text:  #8ea0b1;
    --rule:        #2b3843;
    --accent:      #e2a049;
    --accent-soft: #2e2519;
  }}

  * {{ box-sizing: border-box; }}
  body {{
    margin: 0;
    background: var(--ground);
    color: var(--ink);
    font-family: var(--serif);
    font-size: 18px;
    line-height: 1.62;
    -webkit-font-smoothing: antialiased;
  }}
  .wrap {{
    max-width: var(--wide);
    margin: 0 auto;
    padding: 0 24px 96px;
  }}
  .col {{ max-width: var(--measure); }}

  header.masthead {{
    padding: 72px 0 40px;
    border-bottom: 1px solid var(--rule);
    margin-bottom: 48px;
  }}
  .kicker {{
    font-family: var(--mono);
    font-size: 11.5px;
    letter-spacing: 0.13em;
    text-transform: uppercase;
    color: var(--accent);
    margin: 0 0 18px;
  }}
  h1 {{
    font-size: clamp(30px, 4.4vw, 46px);
    line-height: 1.12;
    font-weight: 600;
    margin: 0 0 22px;
    text-wrap: balance;
    letter-spacing: -0.012em;
  }}
  .standfirst {{
    font-size: 20px;
    line-height: 1.55;
    color: var(--ink-soft);
    margin: 0 0 34px;
  }}

  .facts {{
    display: flex;
    flex-wrap: wrap;
    gap: 0;
    border: 1px solid var(--rule);
    border-radius: 3px;
    overflow: hidden;
    background: var(--surface);
  }}
  .fact {{
    flex: 1 1 150px;
    padding: 15px 20px;
    border-right: 1px solid var(--rule);
  }}
  .fact:last-child {{ border-right: 0; }}
  .fact dt {{
    font-family: var(--mono);
    font-size: 10.5px;
    letter-spacing: 0.11em;
    text-transform: uppercase;
    color: var(--muted-text);
    margin: 0 0 6px;
  }}
  .fact dd {{
    margin: 0;
    font-family: var(--mono);
    font-size: 21px;
    font-variant-numeric: tabular-nums;
    color: var(--ink);
  }}
  .fact dd small {{
    font-family: var(--serif);
    font-size: 14px;
    color: var(--muted-text);
    margin-left: 3px;
  }}

  .lede {{ margin-bottom: 56px; }}
  .lede p {{ margin: 0 0 18px; }}
  .lede p:last-child {{ margin-bottom: 0; }}
  .pull {{
    border-left: 2px solid var(--accent);
    padding-left: 20px;
    margin: 30px 0;
    font-size: 20px;
    line-height: 1.5;
    color: var(--ink);
  }}

  .gallery {{
    display: flex;
    flex-direction: column;
    gap: 60px;
  }}
  .card {{
    margin: 0;
    background: var(--surface);
    border: 1px solid var(--rule);
    border-radius: 4px;
    overflow: hidden;
  }}
  .card-head {{
    padding: 22px 26px 18px;
    display: grid;
    grid-template-columns: 1fr auto;
    align-items: start;
    gap: 6px 16px;
  }}
  .eyebrow {{
    grid-column: 1 / 2;
    font-family: var(--mono);
    font-size: 11px;
    letter-spacing: 0.13em;
    text-transform: uppercase;
    color: var(--accent);
    margin: 0;
  }}
  .card-head h2 {{
    grid-column: 1 / 2;
    font-size: 23px;
    line-height: 1.25;
    font-weight: 600;
    margin: 0;
    text-wrap: balance;
    letter-spacing: -0.008em;
  }}
  .chip {{
    grid-column: 2 / 3;
    grid-row: 1 / 3;
    align-self: center;
    font-family: var(--mono);
    font-size: 10.5px;
    letter-spacing: 0.07em;
    text-transform: uppercase;
    padding: 5px 10px;
    border-radius: 2px;
    white-space: nowrap;
  }}
  .chip-disk {{ background: var(--surface-alt); color: var(--muted-text); }}
  .chip-gbar {{ background: var(--accent-soft); color: var(--accent); }}

  .mount {{
    background: #ffffff;
    border-top: 1px solid var(--rule);
    border-bottom: 1px solid var(--rule);
    overflow-x: auto;
  }}
  .mount img {{
    display: block;
    width: 100%;
    height: auto;
    min-width: 620px;
  }}
  .pending-slot {{
    min-height: 190px;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 10px;
    background: repeating-linear-gradient(
      45deg, #ffffff, #ffffff 11px, #f4f6f8 11px, #f4f6f8 22px);
  }}
  .pending-mark {{
    font-family: var(--mono);
    font-size: 11px;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #8b7a5e;
  }}
  .pending-slot code {{
    font-family: var(--mono);
    font-size: 12.5px;
    color: #7b8794;
  }}
  figcaption {{
    padding: 20px 26px 24px;
    max-width: 72ch;
    color: var(--ink-soft);
    font-size: 16.5px;
    line-height: 1.6;
  }}

  footer {{
    margin-top: 72px;
    padding-top: 26px;
    border-top: 1px solid var(--rule);
    font-family: var(--mono);
    font-size: 12.5px;
    line-height: 1.9;
    color: var(--muted-text);
  }}
  footer b {{ color: var(--ink-soft); font-weight: 500; }}

  @media (max-width: 620px) {{
    body {{ font-size: 17px; }}
    .card-head {{ grid-template-columns: 1fr; }}
    .chip {{ grid-column: 1 / 2; grid-row: auto; justify-self: start; margin-top: 4px; }}
  }}
</style>

<div class="wrap">
  <header class="masthead">
    <div class="col">
      <p class="kicker">design regret &middot; mechanism study</p>
      <h1>Why design regret peaks, then falls, as neighbors accumulate</h1>
      <p class="standfirst">Six views of a single counter-intuitive result: surrounding a wind
        farm with more neighbors eventually makes it matter <em>less</em> whether you knew they
        were coming.</p>
    </div>
    <dl class="facts">
      <div class="fact"><dt>peak regret at</dt><dd>n&nbsp;=&nbsp;{npk}</dd></div>
      <div class="fact"><dt>peak value</dt><dd>{rpk:.1f}<small>% AEP</small></dd></div>
      <div class="fact"><dt>by n = {nlast}</dt><dd>{rlast:.1f}<small>% AEP</small></dd></div>
      <div class="fact"><dt>recoverable share</dt><dd>{rl1:.2f}&nbsp;&rarr;&nbsp;{rl8:.2f}</dd></div>
      <div class="fact"><dt>figures</dt><dd>{n_built}<small>of {len(FIGS)}</small></dd></div>
    </dl>
  </header>

  <section class="col lede">
    <p>Design regret measures the AEP a developer forfeits by laying out a farm in isolation
      when neighbors were coming. It is not a measure of damage. It is the value of foresight —
      and foresight is only worth something if it changes what you would have built.</p>
    <p>That distinction explains the shape of every figure below. Regret factors cleanly into the
      damage a neighbor inflicts and the share of that damage a redesigned layout can dodge. Adding
      farms around a target raises the first term and lowers the second, because each new farm closes
      off another direction the optimizer could have escaped toward. The product peaks somewhere in
      the middle.</p>
    <p class="pull">Perfect knowledge of a threat you cannot dodge is worth nothing. A farm boxed in
      by wake on the directions its energy arrives from is damaged whatever it does, so the layout
      you would build knowing the neighbors and the layout you would build blind converge — and the
      regret between them collapses, even as the total loss is at its worst.</p>
    <p>The obvious reading of that is <em>symmetry</em>: encircle a farm and it has nowhere to go.
      It is also what this study set out to confirm, and it did not survive the controlled test. Holding
      total neighbor capacity and distance fixed while dividing it into more, smaller farms spreads the
      threat around the compass and yet makes escape <em>easier</em>, not harder. The quantity that
      actually governs the recoverable share is how much wake mass sits in the directions carrying
      energy, and how concentrated it is — adding full-size farms piles it on, dividing a fixed
      capacity thins it out.</p>
    <p>The first five figures come from results already on disk. The last two required new
      cluster runs, because they test the account rather than restate it — one by showing the
      turbine movements directly, the other by holding capacity, area and distance fixed so that
      the number and size of the neighbors is what varies.</p>
  </section>

  <div class="gallery">{''.join(cards)}
  </div>

  <footer>
    <b>data</b> &nbsp;ring sweep, n = 1&ndash;8, three wind roses, Bastankhah and TurboPark,
    2D and 10D nominal gaps &middot; FunWake schedule, K = 500 &middot; 50 &times; IEA 15 MW,
    DEI concession polygon<br>
    <b>build</b> &nbsp;<code>scripts/plot_regret_mechanism.py</code>,
    <code>scripts/plot_ring_tier3.py</code> &middot; figures written to
    <code>paper_v3/figures/</code><br>
    <b>status</b> &nbsp;{n_built} of {len(FIGS)} figures built{'' if n_pending == 0 else f' &middot; {n_pending} awaiting gbar array 29051979 (24 jobs)'}
  </footer>
</div>
"""
    OUT.write_text(html)
    kb = len(html.encode()) / 1024
    print(f"Wrote {OUT}  ({kb:.0f} KB, {n_built}/{len(FIGS)} figures embedded)")


if __name__ == "__main__":
    main()
