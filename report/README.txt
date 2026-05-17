Video-Based Pediatric Seizure Detection from Pose Landmark Sequences
====================================================================

Author: Babajide Owoyele
Submission: MaskBench Extension project slot

Contents
--------

INSIDE THE REPORT ZIP (upload to the "Report" field on the portal):

  report.pdf            Full research paper (22 pages). Primary deliverable.
  report.md             Markdown source of the full paper.
  report.tex            LaTeX source of the full paper (auto-generated via pandoc).
  style.css             CSS used to render report.md → report.html → report.pdf.

  report-summary.pdf    Compact 3-page brief.
  report-summary.md     Markdown source of the 3-page brief.
  summary-style.css     CSS for the compact brief.

  README.txt            This file.

SEPARATE UPLOAD (Poster field on the portal — do NOT zip):

  poster.pdf            A0 portrait conference poster (single page).
  poster.html           HTML source of the poster (renders directly in a browser).

SEPARATE UPLOAD (Slides field on the portal — do NOT zip):

  slides.pdf            7-slide widescreen (16:9) deck for a ~5-minute talk.
  slides.html           HTML source of the slide deck.

Funding
-------

This work was supported by the Federal Ministry of Research, Technology and
Space under the funding code "KI-Servicezentrum Berlin-Brandenburg" (16IS22092).
Responsibility for the content remains with the authors.

How to produce the PDF
----------------------

Option A — LaTeX (recommended):
    pdflatex report.tex
    pdflatex report.tex          (run twice so the table of references settles)

Option B — Overleaf:
    Upload report.tex to a blank Overleaf project; compile with pdfLaTeX.

Option C — Pandoc from the Markdown source:
    pandoc report.md -o report.pdf --pdf-engine=xelatex

Code
----

Full source code for the work described in this report is in the same
repository as this report directory. Top-level entry points:

    train_all.py        — 5-fold × 2-architecture training
    main.py             — challenge Docker entry point (.npy arrays)
    predict_video.py    — raw-video inference

See the repo's top-level README.md for a full description of the
pipeline, hardware requirements, and reproduction steps.
