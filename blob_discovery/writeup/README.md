# Pooled Multi-Start Blob Discovery Writeup

This folder contains the LaTeX documentation for the pooled multi-start adversarial blob discovery methodology.

## Key Insight

**Single-shot optimization is insufficient for proper regret estimation.** Apparent regret from single runs conflates optimization artifacts (local minima) with true strategic tradeoffs. The pooled methodology:

1. Runs N multi-start optimizations for both liberal and conservative strategies
2. Cross-evaluates ALL layouts under BOTH scenarios
3. Computes regret against pooled global bests
4. Determines whether a TRUE TRADEOFF exists (different layouts achieve the two global bests)

## Files

- `blob_discovery.tex` - Main LaTeX document
- `discovery_seed0.png` - Results figure (referenced in document)
- `pareto_frontier.png` - Pareto frontier of minimum regrets

## Compilation

To compile the document to PDF:

```bash
cd analysis/blob_discovery/writeup
pdflatex blob_discovery.tex
pdflatex blob_discovery.tex  # Run twice for references
```

Or using latexmk:

```bash
latexmk -pdf blob_discovery.tex
```

## Required LaTeX Packages

- amsmath, amssymb, amsthm
- graphicx
- algorithm, algpseudocode
- booktabs
- hyperref
- cleverref

## Document Overview

The document covers:

1. **Introduction** - Problem setting and design regret definition
2. **Method** - B-spline representation, SDF computation, soft packing, **pooled multi-start optimization**
3. **Experimental Setup** - Configuration parameters
4. **Results** - Pooled optimization results, distinguishing true tradeoffs from artifacts
5. **Discussion** - Why pooled optimization is essential, limitations, future work
6. **Conclusion** - Summary of proper regret estimation methodology
