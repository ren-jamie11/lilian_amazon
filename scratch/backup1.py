import streamlit as st
import pandas as pd
from helpers import *

# --- Streamlit Setup ---
st.set_page_config(page_title="Amazon Sales Analysis", layout="wide")
st.title("🖼️ Amazon Sales Analysis")

ITEM_COLS = ['ASIN', 'SKU', 'URL', '所属类目', '商品标题']
ITEM_COLS_NEW = ['ASIN', 'SKU', 'url', 'category', 'product_title']

df = None
listing_months = None
null_ASINS = []
summary = None

# Uploaders
sales_csv = st.file_uploader("Upload sales csv", type="csv")
if sales_csv is not None:
    # load
    st.session_state['sales'] = pd.read_csv(sales_csv)
    sales = st.session_state['sales']

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

prices_csv = st.file_uploader("Upload prices csv", type="csv")
if prices_csv is not None:
    # load 
    st.session_state['prices'] = pd.read_csv(prices_csv)
    prices = st.session_state['prices']
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



if (sales_csv is not None) and (prices_csv is not None):
    st.write(sales)
    st.write(prices)
    st.write(df)

if summary is not None:
    st.write('Summary')
    st.write(summary)














# # Display DataFrames only if loaded
# if (st.session_state['sales'] is not None) and (st.session_state['prices'] is not None):
#     # df = metadata for each ASIN (excluding price/sales time series)


    

# else:
#     st.info("Please upload sales and prices CSV files before proceeding")