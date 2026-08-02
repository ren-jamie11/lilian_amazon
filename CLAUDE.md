# CLAUDE.md

Guidance for Claude Code working in this repository.

## What this is

A single-page Streamlit dashboard for Amazon market surveillance. The user uploads
one Excel workbook containing monthly sales and price data for a list of ASINs
(typically one product category, e.g. hydrangea trees), and the app aggregates it
into market-size, growth, concentration and competitor views.

The UI is written in **Chinese**; internal identifiers, comments and docstrings are
in English. Keep that split when adding features — user-facing strings (headers,
labels, captions, error messages) should be Chinese.

## Running

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then upload an `.xlsx` in the "上传 Excel" expander and click "Load data".
`example_data/绣球花树——数据.xlsx` is a real sample input.

There are no tests, no linter config and no build step. To verify a change, run
the app and load the example workbook.

## Input data contract

The workbook must have **at least two sheets, order-dependent**:

- **Sheet 0 = sales** (units per month)
- **Sheet 1 = prices** (USD per month; month headers carry a `($)` suffix)

Both sheets are wide: one row per ASIN, one column per month. Required metadata
columns on the sales sheet (`ITEM_COLS` in [app.py](app.py#L40)):
`ASIN, SKU, 品牌, URL, 商品主图, 所属类目, 商品标题, 上架时间` — renamed on load to
`ASIN, SKU, brand, url, image_path, category, product_title, listing_date`.
Month columns look like `2026-07` / `2026-07($)`; `商品主图` is a *local filesystem
path* to a product image, which is why images sometimes fail to render.

`load_data()` validates all of this up front and writes a Chinese message to
`st.session_state['load_error']` rather than raising — preserve that pattern.

## Architecture

Only three Python files, all flat at the repo root:

- **[app.py](app.py)** — everything Streamlit: session-state init, `load_data()`,
  the filter/image-grid widgets, and the three tabs. No analytics live here beyond
  small display helpers.
- **[helpers.py](helpers.py)** — all data cleaning, aggregation, metrics and
  plotting. Imported wholesale via `from helpers import *`, so **every new
  top-level name in helpers.py lands in app.py's namespace** — avoid generic names.
- **[excel.py](excel.py)** — a throwaway two-sheet-preview scratch script. Not
  imported by anything; ignore it.

### Data flow

`load_data()` (on button click) does all the heavy work once and stashes results in
`st.session_state`; the rest of the script re-reads session state on every rerun.

1. Read both sheets → validate → build `df` (per-ASIN metadata).
2. `listing_date` is filled from `上架时间`; where blank, an *implied* listing date
   is derived from the oldest month with non-zero sales.
3. `get_asin_and_months()` strips both sheets down to `ASIN` + month columns
   (dropping the `($)` suffix), `clean_time_series()` coerces `'1,234'` strings to
   numbers and NaN → 0.
4. Listings are bucketed into **cohorts** by listing year via `assign_cohort()`
   (plus `未知` for missing dates). Each cohort gets its own sales/price slice and
   its own summary. See "Cohorts" below.
5. `summarize_price_sales()` produces the per-month `summary` frame:
   `month, total_sales, wavg_price, n_listings, *_pct_change, *_growth_yoy,
   sales_per_listing`.

Key session-state keys: `sales`, `prices`, `df`, `summary`, `pct_changes`,
`reference_month`, `cohort_labels`, `cohort_summaries` (one per cohort, aligned
with `cohort_labels`), `asins_by_cohort`, `cohort_summary`, `load_error`,
`csv_expander`.

### Cohorts

Cohorts are **derived, never hardcoded** — the old fixed
`COHORT_LABELS = ['2023及以前', '2024-2025', '2026及以后']` constant is gone.

- **Anchor** = `get_reference_month(sales)`, the workbook's latest month column
  (falling back to today only if there are none). Anchoring on the data, not the
  clock, means a given file always produces the same split.
- **Count** = 3 when the anchor month `>= COHORT_SPLIT_MONTH` (7), else 4. The two
  or three most recent years are split out individually and everything older
  collapses into a trailing bucket.
- **Labels** come from `get_cohort_labels(reference)` and are **oldest-first**, so
  `cohort_labels[-1]` is always the newest cohort. Anchor 2026-07 →
  `['2024及以前', '2025', '2026及以后']`; anchor 2026-02 →
  `['2023及以前', '2024', '2025', '2026及以后']`.
- `assign_cohort(listing_dates, reference)` requires the anchor — there is
  deliberately no default, since defaulting to today would reintroduce drift.

Anything cohort-aware must be **driven by the `cohort_labels` list**, never by a
fixed count or index: the 总体 dropdowns build their options from it via
`cohort_summary_selector()`, and the 市场份额 tab uses `cohort_labels[-1]` for the
新品 metric and `cohort_labels + [COHORT_UNKNOWN]` for chronological ordering.
A cohort can legitimately be **empty** (a 4-cohort split often has no listings in
the oldest bucket) — `summarize_price_sales()` returns an empty frame,
`plot_ts_two_cols` draws nothing, and callers must handle it.

### Tabs

- **单品 (micro)** — per-ASIN sales time series with an ASIN highlight box, plus a
  price-vs-avg-monthly-sales scatter (optionally colored by cohort).
- **总体 (macro)** — cohort-selectable dual-axis charts for total sales, sales per
  listing and weighted-average price, plus YoY tables for units and revenue over
  1 / 3 / 12-month windows.
- **市场份额 (concentration)** — last-3-month share: brand count, listing count,
  Top 5/10 brand and ASIN share, and stacked share bars by ASIN, brand and cohort.

Below the tabs (always visible) is the 竞争对手分析 section: a sales-cutoff filter,
a keyword/range filter UI, an image grid of matching competitor listings, and a
"fast growing" list.

## Conventions and gotchas

- **Wide format everywhere.** Month columns are sorted `str` keys (`'2026-07'`);
  they sort lexicographically = chronologically. `[c for c in df.columns if c !=
  asin_col]` is the standard idiom for "the month columns".
- **`@st.cache_data` on functions that take DataFrames.** Streamlit hashes the
  frames, so args must stay hashable — don't pass mutable non-DataFrame state into
  cached helpers. `plot_ts_two_cols` mutates its `df` argument in place
  (`df[date_col] = pd.to_datetime(...)`); it's cached, so leave that alone unless
  you also fix the mutation.
- **Month-header parsing is duplicated.** `_month_key()` in app.py tolerates
  headers that openpyxl parsed into datetimes; `filter_date_cols()` in helpers.py
  uses `re.match` and only handles strings. If you touch one, check the other.
- **Prices are per-listing, not per-unit.** `extract_qty()` parses "set of 3" /
  "2-pack" from titles, but the division of price by qty in `load_data()` is
  deliberately commented out. Revenue figures are therefore listing-level GMV.
- **Prices are gap-filled, sales are not.** `backfill_prices()` fills interior
  price gaps with the midpoint of the bracketing months (not a linear
  interpolation), and carries edges outward. Never backfill sales.
- **YoY compares by date, not position.** `yoy_period_growth()` returns `None` if
  the prior-year months are missing, instead of silently comparing wrong months.
  Callers must handle `None` (see `build_yoy_rows`).
- **Empty-input guards matter.** A cohort can legitimately have zero listings;
  `summarize_price_sales()` returns an empty frame in that case and callers check
  `.empty`. Preserve those checks when refactoring.
- **`OUR_BRANDS`** in [app.py](app.py#L453) is a hardcoded list of the user's own
  brands, matched case-insensitively; those ASINs render red in the scatter.
- **Share bars are hand-rolled HTML** (`render_share_bar`, `_share_segment_html`)
  via `st.markdown(..., unsafe_allow_html=True)`, with a fixed `SHARE_PALETTE`.
  In-bar labels are suppressed below `IN_BAR_LABEL_MIN_SHARE` (7%). Colors are
  chosen to stay legible in both Streamlit themes.
- **Charts are matplotlib**, rendered with `st.pyplot(fig)`. Keep new charts
  consistent with the existing style (red = ours/top, gray = background, blue =
  highlighted).
- **Chinese text in matplotlib needs the CJK font fallback** set at the top of
  helpers.py (`plt.rcParams['font.sans-serif']`). Without it, CJK labels — such as
  the cohort names in the scatter legend — render as tofu boxes with a
  "Glyph … missing from current font" warning.
- There is a fair amount of **commented-out code** (old plot versions, the qty
  division, the cohort summary table). It's kept intentionally as reference —
  don't delete it as cleanup unless asked.

## Working style

- Pin dependency versions in `requirements.txt`; the app targets pandas 3.x and
  Streamlit 1.56 (note `width='stretch'` instead of the deprecated
  `use_container_width`).
- Commit messages in this repo are short and informal (e.g. "YoY% table").
- Prefer adding analytics to `helpers.py` and keeping `app.py` to layout and
  wiring.
