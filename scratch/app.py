import streamlit as st
import pandas as pd
from helpers import *

# --- Streamlit Setup ---
st.set_page_config(page_title="Amazon Sales Analysis", layout="wide")
st.title("🖼️ Amazon Sales")

ITEM_COLS = ['ASIN', 'SKU', 'URL', '所属类目', '商品标题']
ITEM_COLS_NEW = ['ASIN', 'SKU', 'url', 'category', 'product_title']

if 'sales' not in st.session_state:
    st.session_state['sales'] = None

if 'prices' not in st.session_state:
    st.session_state['prices'] = None

if 'df' not in st.session_state:
    st.session_state['df'] = None

if 'null_ASINS' not in st.session_state:
     st.session_state['null_ASINS'] = []

if 'summary' not in st.session_state:
    st.session_state['summary'] = None

if "csv_expander" not in st.session_state:
    st.session_state['csv_expander'] = True

if "our_asin" not in st.session_state:
    st.session_state['our_asin'] = ''

def load_data():
    # if sales or price is none...return
    if sales_csv is None:
         st.warning("Please upload sales csv")
         return
        
    if prices_csv is None:
         st.warning("Please upload prices csv")
         return
    
    # load
    sales = pd.read_csv(sales_csv)
    prices = pd.read_csv(prices_csv)

    # retrieve metadata df
    df = sales[ITEM_COLS]
    df.columns = ITEM_COLS_NEW

    # get asin and month cols only
    sales = get_asin_and_months(sales)
    prices = get_asin_and_months(prices)

    # listing date for each asin
    listing_months = get_listing_months(sales)  
    null_ASINS = listing_months[listing_months.listing_date.isna()].ASIN.values
    
    # remove null asins
    df = df[~df.ASIN.isin(null_ASINS)]
    sales = sales[~sales.ASIN.isin(null_ASINS)]
    prices = prices[~prices.ASIN.isin(null_ASINS)]

    # clean
    sales = clean_time_series(sales, int)
    prices = clean_time_series(prices, float)

    # qty
    df = pd.merge(df, listing_months, on = 'ASIN')
    df['qty'] = df['product_title'].apply(extract_qty)  

    # summary
    summary = summarize_price_sales(sales, prices)
    summary['total_sales_pct_change'] = summary['total_sales'].pct_change().round(2)
    summary['n_listings_pct_change'] = summary['n_listings'].pct_change().round(2)

    # store session state
    st.session_state['sales'] = sales
    st.session_state['prices'] = prices
    st.session_state["df"] = df
    st.session_state['summary'] = summary

    st.session_state['csv_expander'] = False

with st.expander("Upload csvs", expanded=st.session_state['csv_expander']):
    sales_csv = st.file_uploader("Sales (销量)", type="csv")
    prices_csv = st.file_uploader("Price (价格)", type="csv")

    load_data_button = st.button("Load data", 
                                 key = "load_data_button",
                                 on_click = load_data)


DATAFRAMES = ["sales", "prices", "df", "summary"]

N_MONTHS = 3
TODAY = get_today_yyyymm()
SALES_CUTOFF_MARGIN = 0.75
GROWTH_CUTOFF = 0.5

if all(st.session_state.get(k) is not None for k in DATAFRAMES):
    sales =  st.session_state['sales']
    prices = st.session_state['prices']
    df = st.session_state["df"]
    summary = st.session_state['summary']

    tabs = st.tabs(["Micro", "Macro"])

    with tabs[0]:  # Micro tab
        st.header("Micro View")

        our_asin = st.text_input(
        "ASIN:",
        value='B0C2HJ2S8F',
        label_visibility="visible",
        placeholder="(e.g. B0C2HJ2S8F)",
        width=300,
        key = 'our_asin'
    )

        c1, c2 = st.columns([6,4])
        with c1:
            # Sales timeseries for a single ASIN
            plot_sales_timeseries(st.session_state['sales'], my_asin=st.session_state['our_asin'])
        
        with c2:
            # Scatter of price vs total sales
            asin_price_sales = scatter_price_vs_sales(
                st.session_state['prices'], 
                st.session_state['sales'], 
                n_months=N_MONTHS, 
                our_asin=st.session_state['our_asin']
            )

    with tabs[1]:  # Macro tab
        st.header("Macro View")

        c3, c4 = st.columns([5,5])
        with c3:
            plot_ts_two_cols(
                st.session_state['summary'], 
                'month', 
                'wavg_price', 
                'n_listings',
                start_date='2022-01', 
                end_date=TODAY
            )
        
        with c4:
            plot_ts_two_cols(
                st.session_state['summary'], 
                'month', 
                'total_sales', 
                'n_listings',
                start_date='2022-01', 
                end_date=TODAY
            )
            
    st.header("Competitor analysis")
    st.markdown("#### Filter by sales")

    # get cutoff qty for our asin
    if our_asin in asin_price_sales.ASIN.values:
        # print(f"{our_asin} is in asin_price_sales")
        our_asin_qty = asin_price_sales[asin_price_sales.ASIN == our_asin].monthly_sales.values[0]
    else:
        # print(f"{our_asin} is NOT in asin_price_sales")
        our_asin_qty = 0

    cutoff_qty_input = st.text_input("平均月销量", value=int(our_asin_qty * SALES_CUTOFF_MARGIN), width = 150)

    # validate input
    try:
        cutoff_qty = float(cutoff_qty_input)
    except ValueError:
        st.error("Please enter a valid integer for cutoff quantity.")
        cutoff_qty = None

    # only calculate rival_asins if cutoff_qty is valid
    if isinstance(cutoff_qty, (int, float)):
        rival_asins = asin_price_sales[asin_price_sales.monthly_sales > cutoff_qty * SALES_CUTOFF_MARGIN]
        rival_asins = rival_asins.merge(
            st.session_state['df'][['ASIN', 'url', 'product_title', 'listing_date']], 
            on='ASIN'
        )
    else:
        # return empty DataFrame with same columns
        rival_asins = pd.DataFrame(columns=['ASIN', 'monthly_sales', 'url', 'product_title', 'listing_date'])
            
    st.write(rival_asins)


    st.markdown("#### Filter by fast-growing")

    growth_cutoff = st.text_input("三月销量增长率(%)", value=int(GROWTH_CUTOFF*100), width = 150)

    try:
        growth_cutoff = float(growth_cutoff) / 100
    except ValueError:
        st.error("Please enter a valid integer for cutoff quantity.")
        growth_cutoff = GROWTH_CUTOFF

    fast_growing_asins = get_fast_growing_asins(sales, asin_price_sales, growth_cutoff = growth_cutoff, sales_cutoff = 200)
    fast_growing_asins = fast_growing_asins.merge(df[['ASIN', 'url', 'product_title', 'listing_date']], on = 'ASIN')
    st.write(fast_growing_asins)