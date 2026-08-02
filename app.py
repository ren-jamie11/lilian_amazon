import streamlit as st
import pandas as pd
from helpers import *

import numpy as np
from PIL import UnidentifiedImageError

if 'sales' not in st.session_state:
    st.session_state['sales'] = None

if 'prices' not in st.session_state:
    st.session_state['prices'] = None

if 'df' not in st.session_state:
    st.session_state['df'] = None

if 'summary' not in st.session_state:
    st.session_state['summary'] = None

if 'pct_changes' not in st.session_state:
    st.session_state['pct_changes'] = None

if "csv_expander" not in st.session_state:
    st.session_state['csv_expander'] = True

if "load_error" not in st.session_state:
    st.session_state['load_error'] = None

# --- Streamlit Setup ---
st.set_page_config(page_title="Amazon Sales Analysis", layout="wide")
st.header("📈 Amazon 产品生命周期分析")

ITEM_COLS = ['ASIN', 'SKU', '品牌','URL', '商品主图', '所属类目', '商品标题', '上架时间']
ITEM_COLS_NEW = ['ASIN', 'SKU', 'brand','url', 'image_path','category', 'product_title', 'listing_date']
DISPLAY_COLS = ['ASIN', 'brand','url', 'image_path','product_title', 'listing_date']

# most listings the image grid will render before truncating
IMAGE_GRID_LIMIT = 100


def _month_key(col):
    """Return a normalized 'YYYY-MM' string if `col` is a month-like column
    label, otherwise None.

    Handles the normal case where headers are strings like '2026-07' (with or
    without a trailing price marker such as '($)'), and the case where Excel /
    openpyxl parsed a 'YYYY-MM' header into a date/datetime/Timestamp object.
    """
    # Date-like header (has year/month attributes but isn't a plain string)
    if not isinstance(col, str) and hasattr(col, 'year') and hasattr(col, 'month'):
        try:
            return f"{int(col.year):04d}-{int(col.month):02d}"
        except Exception:
            return None
    s = str(col).strip()
    if len(s) >= 7 and s[4] == '-' and s[:4].isdigit() and s[5:7].isdigit():
        return f"{s[:4]}-{s[5:7]}"
    return None


def load_data():
    # --- Structural validation: hard-fail with a clear message ---
    if excel_file is None:
        st.session_state['load_error'] = "请先上传一个 Excel 文件。"
        return

    try:
        xl = pd.ExcelFile(excel_file, engine='openpyxl')
    except Exception as e:
        st.session_state['load_error'] = (
            f"无法读取该文件，请确认它是有效的 .xlsx 文件。（{e}）"
        )
        return

    if len(xl.sheet_names) < 2:
        st.session_state['load_error'] = (
            f"文件需要至少两个工作表（第一个为销量，第二个为价格），"
            f"但只找到 {len(xl.sheet_names)} 个。"
        )
        return

    try:
        sales = pd.read_excel(xl, sheet_name=0, engine='openpyxl')   # First sheet
        prices = pd.read_excel(xl, sheet_name=1, engine='openpyxl')  # Second sheet
    except Exception as e:
        st.session_state['load_error'] = f"读取工作表时出错：{e}"
        return

    # All expected metadata columns must be present on the sales sheet
    missing_cols = [c for c in ITEM_COLS if c not in sales.columns]
    if missing_cols:
        st.session_state['load_error'] = (
            "销量工作表缺少必要的列：" + "、".join(map(str, missing_cols))
        )
        return

    # The price sheet must at least have ASIN to join on
    if 'ASIN' not in prices.columns:
        st.session_state['load_error'] = "价格工作表缺少必要的列：ASIN"
        return

    # At least one month column (e.g. 2026-07) must exist
    date_columns = [col for col in sales.columns if _month_key(col) is not None]
    if not date_columns:
        st.session_state['load_error'] = (
            "未在销量工作表中找到任何月份列（应形如 2026-07）。"
        )
        return

    # Passed validation — clear any previous error
    st.session_state['load_error'] = None

    # retrieve metadata df (copy to avoid chained-assignment warnings)
    df = sales[ITEM_COLS].copy()
    df.columns = ITEM_COLS_NEW

    # drop rows with no ASIN — nothing downstream can key on them
    df = df[df['ASIN'].notna()].copy()

    # get listings with data (coerce bad/blank dates to NaT rather than crashing)
    df['listing_date'] = pd.to_datetime(df['listing_date'], errors='coerce')

    def find_implied_listing_date(row):
        for col in reversed(date_columns):  # Start from oldest date
            value = row[col]
            # Check if value is not NaN and is greater than 0
            if pd.notna(value) and value > 0:
                return _month_key(col)
        return None  # If no valid date found

    df['implied_listing_date'] = sales.apply(find_implied_listing_date, axis=1)

    df['implied_listing_date'] = pd.to_datetime(
        df['implied_listing_date'].apply(lambda x: f"{x}-01" if pd.notna(x) else None),
        format='%Y-%m-%d',
        errors='coerce'
    )

    df['listing_date'] = df['listing_date'].fillna(df['implied_listing_date'])
    df = df.drop(columns='implied_listing_date')

    asins = df.ASIN.values.tolist()

    # get asin and month cols only
    sales = get_asin_and_months(sales)
    prices = get_asin_and_months(prices)
    
    # remove null asins
    sales = sales[sales.ASIN.isin(asins)]
    prices = prices[prices.ASIN.isin(asins)]

    # clean
    sales = clean_time_series(sales, int)
    prices = clean_time_series(prices, float)

    # qty
    df['qty'] = df['product_title'].apply(extract_qty)  
    merged = prices.merge(df[['ASIN', 'qty']], on='ASIN', how='left')
    # month_cols = [c for c in merged.columns if c not in ['ASIN', 'qty']]
    # merged[month_cols] = merged[month_cols].div(merged['qty'], axis=0)
    prices = merged.drop(columns=['qty'])

    # by cohort — anchored to the workbook's latest month, so the number of
    # cohorts (3 or 4) and their labels come from the data, not the clock
    reference_month = get_reference_month(sales)
    cohort_labels = get_cohort_labels(reference_month)
    cohort = assign_cohort(df['listing_date'], reference_month)

    def get_price_sales_by_cohort(df, sales, prices):
        asins_list = df.ASIN.values.tolist()
        sales_cohort = sales[sales.ASIN.isin(asins_list)]
        prices_cohort = prices[prices.ASIN.isin(asins_list)]
        return sales_cohort, prices_cohort, asins_list

    cohort_dfs = [df[cohort == label] for label in cohort_labels]
    cohort_slices = [get_price_sales_by_cohort(d, sales, prices) for d in cohort_dfs]

    st.session_state['reference_month'] = reference_month
    st.session_state['cohort_labels'] = cohort_labels
    st.session_state['asins_by_cohort'] = [asins for _, _, asins in cohort_slices]

    # summary
    summary = summarize_price_sales(sales, prices, df)
    pct_changes = summary[['month','sales_growth_yoy', 'n_listings_growth_yoy']]
    pct_changes['month'] = pd.to_datetime(pct_changes['month']).dt.strftime('%Y-%m')
    pct_changes = pct_changes.dropna()
    pct_changes = pct_changes.set_index("month").T

    # total sales graph for each cohort
    st.session_state['cohort_summaries'] = [
        summarize_price_sales(sales_c, prices_c, df_c)
        for (sales_c, prices_c, _), df_c in zip(cohort_slices, cohort_dfs)
    ]

    # store session state
    st.session_state['sales'] = sales
    st.session_state['prices'] = prices
    st.session_state["df"] = df
    st.session_state['summary'] = summary
    st.session_state['pct_changes'] = pct_changes

    st.session_state['csv_expander'] = False

with st.expander("上传 Excel", expanded=st.session_state['csv_expander']):    
    excel_file = st.file_uploader("上传 Excel", type="xlsx")
    load_data_button = st.button("Load data", key="load_data_button", on_click=load_data)

if st.session_state.get('load_error'):
    st.error(st.session_state['load_error'])


def drop_stale_selection(key, options):
    """Forget a session value that is no longer one of `options`.

    Cohort labels and brand names are derived from the workbook, so a newly
    loaded file can leave session_state holding a value Streamlit no longer
    offers — which it raises on.
    """
    if key in st.session_state and st.session_state[key] not in options:
        del st.session_state[key]


def rival_price_slider(rival_df, key='rival_price'):
    """Whole-dollar price range slider over `rival_df`.

    Returns the (low, high) tuple, or None when there is nothing to slide over
    — an empty rival set, in which case st.slider's bounds would be degenerate.
    """
    prices = pd.to_numeric(rival_df['price'], errors='coerce').dropna()
    if prices.empty:
        st.caption("价格")
        st.caption("暂无数据")
        return None

    pmin = float(np.floor(prices.min()))
    pmax = float(np.ceil(prices.max()))
    if pmax <= pmin:
        # every listing at the same price — widen so the slider has a range
        pmax = pmin + 1.0

    # the bounds move with 最少平均月销量, so clamp the stored selection into
    # them rather than letting Streamlit see an out-of-range value
    if key in st.session_state:
        low, high = st.session_state[key]
        st.session_state[key] = (
            min(max(low, pmin), pmax),
            max(min(high, pmax), pmin),
        )

    return st.slider("价格", pmin, pmax, (pmin, pmax), step=1.0, key=key)


def display_images(trimmed_df, n_display = IMAGE_GRID_LIMIT):
        if len(trimmed_df) > n_display:
            trimmed_sample = trimmed_df.iloc[:n_display, :]
        else:
            trimmed_sample = trimmed_df.copy()

        st.write("")  # spacing

        # ---------- Image Display (full screen) ----------


        def to_str(val):
            if isinstance(val, (list, set, tuple)):
                return ", ".join(map(str, val))
            if isinstance(val, np.ndarray):
                return ", ".join(map(str, val.tolist()))
            return str(val)

        grid_cols = st.columns(3)

        for idx, (_, row) in enumerate(trimmed_sample.iterrows()):
            with grid_cols[idx % 3]:
                url = row['url']
                img_path = row["image_path"]
                product_title = to_str(row.get("product_title", []))
                listing_date = row.get("listing_date", []).strftime("%Y-%m")
                
                try:
                    st.image(img_path)
                    st.caption(url)
                    st.caption(product_title)

                    
                    st.markdown(
                        f"""
                        **ASIN:** {to_str(row.get("ASIN", []))}  
                        **Brand:** {to_str(row.get("brand", []))}  
                        **Listing date:** {listing_date}  
                        **Price:** {to_str(row.get("price", []))}  
                        **Monthly sales:** {to_str(row.get("monthly_sales", []))}  
                        """
                    )

                except (FileNotFoundError, UnidentifiedImageError, OSError):
                    st.warning(f"⚠️ Could not load image: {img_path}")


DATAFRAMES = ["sales", "prices", "df", "summary"]
OUR_BRANDS = ['briful', 'Hollyone', 'Dilatata','Villa Como', 'Arborus', 'Oairse', 'Nature Crafted']

N_MONTHS = 3
TODAY = get_today_yyyymm()
SALES_CUTOFF_MARGIN = 1
GROWTH_CUTOFF = 0.5

FILTER_ALL = '全部'
TOP_BRAND_N = 5

if all(st.session_state.get(k) is not None for k in DATAFRAMES):
    sales =  st.session_state['sales']
    prices = st.session_state['prices']
    df = st.session_state["df"]
    summary = st.session_state['summary']
    pct_changes = st.session_state['pct_changes']
    reference_month = st.session_state['reference_month']
    cohort_labels = st.session_state['cohort_labels']
    cohort_summaries = st.session_state['cohort_summaries']

    def extract_cohort_row(label, summary):
        if summary.empty:
            return None
        total_sales = summary['total_sales'].tail(3).sum()
        n_listings = summary['n_listings'].replace(0, pd.NA).dropna().iloc[-1] if not summary['n_listings'].replace(0, pd.NA).dropna().empty else pd.NA
        return (label, total_sales, n_listings)

    rows = [
        r for r in (
            extract_cohort_row(label, cohort_summary)
            for label, cohort_summary in zip(cohort_labels, cohort_summaries)
        )
        if r is not None  # drop cohorts with no listings (avoids shape mismatch)
    ]

    st.session_state['cohort_summary'] = (
        pd.DataFrame(rows, columns=['组', '最近3月销量', '产品数量'])
        .dropna()
        .assign(平均月销量=lambda d: (d['最近3月销量'] / d['产品数量'] / 3).round().astype(int))
    )

    OUR_BRAND_ASINS = df[df.brand.str.lower().isin([b.lower() for b in OUR_BRANDS])].ASIN.values.tolist()
    
    # for s in cohort_summaries:
    #     st.write(s)

    tabs = st.tabs(["单品", "总体", "市场份额"])

    with tabs[0]:  # Micro tab
        st.markdown("#### 单品分析")

        highlight_asin = st.text_input(
        "ASIN:",
        label_visibility="visible",
        placeholder="(e.g. B0C2HJ2S8F)",
        width=300,
        key = 'highlight_asin'
    )

        c1, c2 = st.columns([6,4])
        with c1:
            # Sales timeseries for a single ASIN
            plot_sales_timeseries(
                st.session_state['sales'], 
                my_asins=OUR_BRAND_ASINS, 
                highlighted_asin=st.session_state['highlight_asin']
            )

        
        with c2:
            if 'show_cohort' not in st.session_state:
                st.session_state['show_cohort'] = False

            asin_price_sales = scatter_price_vs_sales(
                st.session_state['prices'], 
                st.session_state['sales'], 
                n_months=N_MONTHS, 
                our_asins=OUR_BRAND_ASINS,
                cohort_asins=st.session_state['asins_by_cohort'] if st.session_state['show_cohort'] else None,
                cohort_labels=cohort_labels
            )

            st.toggle("Show cohort", key='show_cohort')

    def cohort_summary_selector(key, label="Cohort"):
        """Selectbox over 总 + the dynamic cohort labels; returns the chosen frame."""
        options = ['总'] + cohort_labels
        drop_stale_selection(key, options)

        st.selectbox(label, options=options, key=key)
        idx = options.index(st.session_state[key])
        selected = summary if idx == 0 else cohort_summaries[idx - 1]

        # a cohort can legitimately have no listings; plot_ts_two_cols draws
        # nothing for an empty frame, so say so rather than leaving a blank gap
        if selected.empty:
            st.caption("该组暂无数据")

        return selected

    with tabs[1]:  # Macro tab
        c3, c4 = st.columns([5,5])
        st.write(st.session_state['summary'])
        with c3:
            st.markdown("#### 平均月销量")
            plot_ts_two_cols(
                cohort_summary_selector('selected_cohort_1'),
                'month',
                'sales_per_listing',
                'n_listings',
                start_date='2022-01',
                end_date=TODAY
            )

            st.write('')
            st.markdown("#### 平均价")
            plot_ts_two_cols(
                cohort_summary_selector('selected_cohort_2'),
                'month',
                'wavg_price',
                'n_listings',
                start_date='2022-01',
                end_date=TODAY
            )


        with c4:
            st.markdown("#### 总销量")
            plot_ts_two_cols(
                cohort_summary_selector('selected_cohort'),
                'month',
                'total_sales',
                'n_listings',
                start_date='2022-01',
                end_date=TODAY
            )

            st.write('')

            def build_yoy_rows(monthly, format_value):
                """One row per horizon; skips horizons the data can't support."""
                rows = []
                for label, n in [('最新月', 1), ('最近3个月', 3), ('最近12个月', 12)]:
                    period = yoy_period_growth(monthly, n)
                    if period is None:
                        continue
                    start, end = period['window']
                    if n == 1:
                        window = start.strftime('%Y-%m')
                    elif start.year == end.year:
                        window = f"{start:%Y-%m}~{end:%m}"
                    else:
                        window = f"{start:%Y-%m}~{end:%Y-%m}"
                    rows.append({
                        '期间': f"{label} {window}",
                        '本期': format_value(period['current']),
                        '去年同期': format_value(period['prior']),
                        '同比': format_growth_pct(period['growth']),
                    })
                return rows

            sales_rows = build_yoy_rows(get_monthly_sales(sales), lambda v: f"{v:,.0f}")
            revenue_rows = build_yoy_rows(
                get_monthly_revenue(sales, prices), format_usd_compact
            )

            if sales_rows or revenue_rows:
                st.markdown("#### 市场增速（销量）")
                st.dataframe(pd.DataFrame(sales_rows), hide_index=True, width='stretch')

                st.write('')
                st.markdown("#### 市场增速（销售额）")
                st.dataframe(pd.DataFrame(revenue_rows), hide_index=True, width='stretch')

                st.caption("与去年同期对比（全市场，不受上方 Cohort 影响）")
            else:
                st.markdown("#### 市场增速")
                st.caption("数据不足，无法计算同比（需至少13个月）")


    with tabs[2]:  # Market share / concentration tab
        st.markdown("#### 市场份额 (最近3月)")

        recent, recent_months = get_recent_sales(sales, n_months=N_MONTHS)
        revenue, _ = get_recent_revenue(sales, prices, n_months=N_MONTHS)
        total_revenue = float(revenue['recent_revenue'].sum()) if not revenue.empty else 0.0

        # one row per listing, with its brand, cohort and recent sales
        share_df = df[['ASIN', 'brand', 'listing_date']].copy()
        share_df['brand_key'], brand_names = normalize_brands(share_df['brand'])
        share_df['cohort'] = assign_cohort(share_df['listing_date'], reference_month)
        share_df = share_df.merge(recent, on='ASIN', how='left')
        share_df['recent_sales'] = share_df['recent_sales'].fillna(0)

        total_sales = float(share_df['recent_sales'].sum())
        n_listings = len(share_df)
        n_brands = share_df['brand_key'].nunique()
        newest_listings = int((share_df['cohort'] == cohort_labels[-1]).sum())

        if recent_months:
            st.caption(
                f"总销量 {total_sales:,.0f} 件 · 总销售额 {format_usd_compact(total_revenue)}"
                
            )

        def _share_of_top(labels, values, n):
            top, _, total = top_n_shares(labels, values, top_n=n)
            if total <= 0:
                return None
            return top['share'].sum()

        def _pct_text(value):
            return "—" if value is None else f"{value:.0f}%"

        brand_top5 = _share_of_top(share_df['brand_key'], share_df['recent_sales'], 5)
        brand_top10 = _share_of_top(share_df['brand_key'], share_df['recent_sales'], 10)
        asin_top5 = _share_of_top(share_df['ASIN'], share_df['recent_sales'], 5)
        asin_top10 = _share_of_top(share_df['ASIN'], share_df['recent_sales'], 10)

        k1, k2, k3, k4 = st.columns(4)
        with k1:
            st.metric("品牌数", f"{n_brands:,}")
        with k2:
            st.metric("商品数", f"{n_listings:,}")
            if n_listings:
                st.caption(
                    f"{cohort_labels[-1]}新品: "
                    f"{newest_listings / n_listings * 100:.0f}% ({newest_listings}个)"
                )
        with k3:
            st.metric("Top 5 品牌份额", _pct_text(brand_top5))
            st.caption(f"Top 10: {_pct_text(brand_top10)}")
        with k4:
            st.metric("Top 5 单品份额", _pct_text(asin_top5))
            st.caption(f"Top 10: {_pct_text(asin_top10)}")

        st.write("")

        if total_sales <= 0:
            st.info("最近3个月无销量数据")
        else:
            # --- top single products ---
            st.markdown("#### 单品集中度")
            asin_labels = share_df['brand_key'].map(brand_names) + " · " + share_df['ASIN']
            top_asins, other_asins, _ = top_n_shares(
                asin_labels, share_df['recent_sales'], top_n=5, other_label='其他单品'
            )
            render_share_bar(
                [
                    (row['label'], row['share'], SHARE_PALETTE[i])
                    for i, row in top_asins.iterrows()
                ],
                other=other_asins,
            )

            st.write("")
            st.write("")

            # --- top brands ---
            st.markdown("#### 品牌集中度")
            top_brands, other_brands, _ = top_n_shares(
                share_df['brand_key'].map(brand_names),
                share_df['recent_sales'],
                top_n=5,
                other_label='其他品牌',
            )
            render_share_bar(
                [
                    (row['label'], row['share'], SHARE_PALETTE[i])
                    for i, row in top_brands.iterrows()
                ],
                other=other_brands,
            )

            st.write("")
            st.write("")

            # --- listing age cohorts (chronological, no remainder) ---
            st.markdown("#### 上架时间分布")
            cohort_order = [c for c in cohort_labels + [COHORT_UNKNOWN]
                            if (share_df['cohort'] == c).any()]
            cohort_sales = share_df.groupby('cohort')['recent_sales'].sum()
            cohort_counts = share_df['cohort'].value_counts()
            render_share_bar([
                (
                    f"{c} ({int(cohort_counts[c])}个)",
                    cohort_sales.get(c, 0) / total_sales * 100,
                    SHARE_PALETTE[i] if i < len(SHARE_PALETTE) else SHARE_OTHER_COLOR,
                )
                for i, c in enumerate(cohort_order)
            ])

            st.write("")

    st.write("")
    st.markdown("#### 竞争对手分析")

    # # get cutoff qty for our asin
    # if our_asin in asin_price_sales.ASIN.values:
    #     # st.write(f"{our_asin} is in asin_price_sales")
    #     our_asin_qty = asin_price_sales[asin_price_sales.ASIN == our_asin].monthly_sales.values[0]
    # else:   
    #     # st.write(f"{our_asin} is NOT in asin_price_sales")
    #     our_asin_qty = 0 

    cutoff_qty_input = st.text_input("最少平均月销量", value=0, width = 150)

    # validate input
    try:
        cutoff_qty = float(cutoff_qty_input)
    except ValueError:
        st.error("Please enter a valid integer for cutoff quantity.")
        cutoff_qty = None

    # per-ASIN cohort/brand attributes + the market's top brands, so the three
    # filters below can be built without re-deriving any of it here
    rival_attrs, top_brand_keys, rival_brand_names = competitor_filter_options(
        df, sales, reference_month, n_months=N_MONTHS, top_n=TOP_BRAND_N
    )

    # only calculate rival_asins if cutoff_qty is valid
    if isinstance(cutoff_qty, (int, float)):
        rival_asins = asin_price_sales[asin_price_sales.monthly_sales >= cutoff_qty * SALES_CUTOFF_MARGIN]
        rival_asins = rival_asins.merge(
            st.session_state['df'][DISPLAY_COLS],
            on='ASIN'
        )
        rival_asins = rival_asins.merge(rival_attrs, on='ASIN', how='left')

    else:
        # return empty DataFrame with same columns
        rival_asins = pd.DataFrame(
            columns=['ASIN', 'price', 'total_sales', 'monthly_sales']
            + DISPLAY_COLS[1:] + ['cohort', 'brand_key']
        )

    ST_COLS = ['ASIN', 'brand','price', 'monthly_sales', 'listing_date','url']

    # --- filters: cohort / brand / price, always visible, AND-combined ---
    # all three are built off the post-cutoff rival set rather than off each
    # other, so the widgets never shift around while you're using them
    cohort_options = [FILTER_ALL] + cohort_labels          # 未知 deliberately not offered
    brand_options = [FILTER_ALL] + [rival_brand_names[k] for k in top_brand_keys]

    drop_stale_selection('rival_cohort', cohort_options)
    drop_stale_selection('rival_brand', brand_options)

    f1, f2, f3 = st.columns(3)
    with f1:
        st.selectbox("上架时间", options=cohort_options, key='rival_cohort')
    with f2:
        st.selectbox("品牌", options=brand_options, key='rival_brand')
    with f3:
        price_range = rival_price_slider(rival_asins)

    trimmed_df = rival_asins
    if st.session_state['rival_cohort'] != FILTER_ALL:
        trimmed_df = trimmed_df[trimmed_df['cohort'] == st.session_state['rival_cohort']]
    if st.session_state['rival_brand'] != FILTER_ALL:
        # match on the normalized key, not the raw brand string, which carries
        # case and whitespace variants of the same brand
        selected_key = top_brand_keys[brand_options.index(st.session_state['rival_brand']) - 1]
        trimmed_df = trimmed_df[trimmed_df['brand_key'] == selected_key]
    if price_range is not None:
        trimmed_df = trimmed_df[trimmed_df['price'].between(*price_range)]

    trimmed_df = trimmed_df.sort_values(by = 'monthly_sales', ascending = False )
    # st.dataframe(trimmed_df[ST_COLS])

    # DISTRIBUTION OF PRICE
    if not trimmed_df.empty:
        avg_price = trimmed_df['price'].mean()
    # st.write(avg_price)


    # plot_price_histogram(trimmed_df)

    if trimmed_df.empty:
        st.info("没有符合条件的商品")
    else:
        n_matched = len(trimmed_df)
        # display_images caps at 100 — say so rather than truncating silently
        st.caption(
            f"共 {n_matched} 个商品"
            if n_matched <= IMAGE_GRID_LIMIT
            else f"共 {n_matched} 个商品（显示前{IMAGE_GRID_LIMIT}个）"
        )
        display_images(trimmed_df, n_display=IMAGE_GRID_LIMIT)

    st.markdown("#### 快速增长的")

    growth_cutoff = st.text_input("三月销量增长率(%)", value=int(GROWTH_CUTOFF*100), width = 150)

    try:
        growth_cutoff = float(growth_cutoff) / 100
    except ValueError:
        st.error("Please enter a valid integer for cutoff quantity.")
        growth_cutoff = GROWTH_CUTOFF

    fast_growing_asins = get_fast_growing_asins(sales, asin_price_sales, growth_cutoff = growth_cutoff, sales_cutoff = 200)
    fast_growing_asins = fast_growing_asins.merge(df[DISPLAY_COLS], on = 'ASIN')
    st.write(fast_growing_asins[ST_COLS])


    
    display_images(fast_growing_asins)