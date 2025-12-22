import streamlit as st
import pandas as pd
from helpers import *

# --- Streamlit Setup ---
st.set_page_config(page_title="Amazon Sales Analysis", layout="wide")
st.title("🖼️ Amazon Sales")

ITEM_COLS = ['ASIN', 'SKU', 'URL', '所属类目', '商品标题']
ITEM_COLS_NEW = ['ASIN', 'SKU', 'url', 'category', 'product_title']


st.session_state['data_loaded'] = False 
st.session_state['csv_expander'] = True

with st.expander("Upload csvs", expanded=st.session_state['csv_expander']):
    sales_csv = st.file_uploader("Sales (销量)", type="csv")
    prices_csv = st.file_uploader("Price (价格)", type="csv")

if not st.session_state['data_loaded']:

    sales = None
    prices = None
    df = None
    listing_months = None

    if sales_csv is not None:
        st.write("SALES AGAIN")
        # load
        st.session_state['sales'] = pd.read_csv(sales_csv)
        sales = st.session_state['sales']
        # st.success(f"Successfully loaded sales csv ({len(sales)} items) ")

        # load df
        df = sales[ITEM_COLS]
        df.columns = ITEM_COLS_NEW

        # get asin and month cols only
        sales = get_asin_and_months(sales)

        # get listing date for each asin
        listing_months = get_listing_months(sales)  
        null_ASINS = listing_months[listing_months.listing_date.isna()].ASIN.values

        # clean
        sales = clean_time_series(sales, int)

    if prices_csv is not None:
        # load 
        st.session_state['prices'] = pd.read_csv(prices_csv)
        prices = st.session_state['prices']
        # st.success(f"Successfully loaded prices csv ({len(prices)} items) ")

        prices = get_asin_and_months(prices)

        # clean 
        prices = clean_time_series(prices, float)

    if (df is not None) and (listing_months is not None) and (prices is not None):
        # remove null asins
        df = df[~df.ASIN.isin(null_ASINS)]
        sales = sales[~sales.ASIN.isin(null_ASINS)]
        prices = prices[~prices.ASIN.isin(null_ASINS)]

        # extract qty
        df['qty'] = df['product_title'].apply(extract_qty)  

        # prepare summary
        summary = summarize_price_sales(sales, prices)
        summary['total_sales_pct_change'] = summary['total_sales'].pct_change().round(2)
        summary['n_listings_pct_change'] = summary['n_listings'].pct_change().round(2)

        # update app state
        st.session_state['data_loaded'] = True
        st.session_state['csv_expander'] = False

    else:
        st.info("Please upload sales and prices CSV files before proceeding")


def inspect_raw_dfs():
    st.write('Sales')
    st.dataframe(sales)

    st.write('Prices')
    st.dataframe(prices)

    st.write('df')
    st.dataframe(df)

    st.write('summary')
    st.dataframe(summary)

if st.session_state['data_loaded']:
    N_MONTHS = 3
    TODAY = get_today_yyyymm()
    
    our_asin = st.text_input(
        "ASIN:",
        value='B0C2HJ2S8F',
        label_visibility="visible",
        placeholder="Please enter the ASIN you want to analyze (e.g. 'B0C2HJ2S8F)",
        width=300
    )
    
    cutoff_qty_input = st.text_input("月销量", value="0")
    # validate input
    try:
        cutoff_qty = int(cutoff_qty_input)
    except ValueError:
        st.error("Please enter a valid integer for cutoff quantity.")
        cutoff_qty = None


    if our_asin and len(our_asin) > 0:
        # Micro
        plot_sales_timeseries(sales, my_asin = our_asin)
        asin_price_sales = scatter_price_vs_sales(prices, sales, n_months = N_MONTHS, our_asin = our_asin)

        # Macro 
        plot_ts_two_cols(summary, 'month', 'wavg_price', 'n_listings' ,
                         start_date = '2022-01', end_date = TODAY)
        
        plot_ts_two_cols(summary, 'month', 'total_sales', 'n_listings' ,
                         start_date = '2022-01', end_date = TODAY)


    # with st.expander("Graphs"):


    # plot_ts_two_cols(summary, 'month', 'wavg_price', 'n_listings',
    #                  start_date = '2023-01', end_date = '2025-11', line_value = None)