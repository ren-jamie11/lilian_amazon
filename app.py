import streamlit as st
import pandas as pd
from helpers import *

import numpy as np
from PIL import UnidentifiedImageError

from pandas.api.types import (
    is_categorical_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_object_dtype,
)

# --- Streamlit Setup ---
st.set_page_config(page_title="Amazon Sales Analysis", layout="wide")
st.title("🖼️ Amazon Sales")

ITEM_COLS = ['ASIN', 'SKU', '品牌','URL', '商品主图', '所属类目', '商品标题', '上架时间']
ITEM_COLS_NEW = ['ASIN', 'SKU', 'brand','url', 'image_path','category', 'product_title', 'listing_date']
DISPLAY_COLS = ['ASIN', 'brand','url', 'image_path','product_title', 'listing_date']

if 'sales' not in st.session_state:
    st.session_state['sales'] = None

if 'prices' not in st.session_state:
    st.session_state['prices'] = None

if 'df' not in st.session_state:
    st.session_state['df'] = None

if 'summary' not in st.session_state:
    st.session_state['summary'] = None

if "csv_expander" not in st.session_state:
    st.session_state['csv_expander'] = True

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

    # get listings with data
    df['listing_date'] = pd.to_datetime(df['listing_date'])
    df = df[~df.listing_date.isna()]
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
    month_cols = [c for c in merged.columns if c not in ['ASIN', 'qty']]
    merged[month_cols] = merged[month_cols].div(merged['qty'], axis=0)
    prices = merged.drop(columns=['qty'])

    # summary
    summary = summarize_price_sales(sales, prices, df)
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


# --- Callbacks ---
def add_keyword_1(column):
    kw = st.session_state[f"new_keyword_{column}"].strip()
    if kw:
        st.session_state[f"keywords_{column}"].add(kw)
    st.session_state[f"new_keyword_{column}"] = ""  # clear input
    # ensure the new keyword appears selected
    st.session_state[f"selected_keywords_{column}"] = sorted(st.session_state[f"keywords_{column}"])

def on_multiselect_change(column):
    selected = set(st.session_state[f"keyword_multiselect_{column}"])
    # Remove anything unselected
    st.session_state[f"keywords_{column}"].intersection_update(selected)
    # Sync selected list
    st.session_state[f"selected_keywords_{column}"] = sorted(st.session_state[f"keywords_{column}"])


def filter_rows_by_all_keywords(df, col, keywords):
    if not keywords or len(keywords) == 0:
        return df
    
    def contains_all_keywords(cell):
        # Handle NaN or None
        if cell is None:
            return False

        # Handle numpy arrays, lists, tuples, sets uniformly
        if isinstance(cell, (np.ndarray, list, tuple, set)):
            cell_iterable = cell
        else:
            # Try to detect NaN scalars
            if pd.isna(cell):
                return False
            cell_iterable = [cell]

        # Lowercase all items for case-insensitive comparison
        cell_strs = [str(item).lower() for item in cell_iterable]

        # Check if every keyword appears in any of the cell strings
        return all(
            any(keyword.lower() in s for s in cell_strs)
            for keyword in keywords
        )

    mask = df[col].apply(contains_all_keywords)
    return df[mask]

def filter_rows_by_exact_keywords(df, col, keywords):
    if not keywords or len(keywords) == 0:
        return df

    target_set = set(k.lower() for k in keywords)

    def matches_exact_keywords(cell):
        # Handle NaN or None
        if cell is None:
            return False

        # Handle list-like cells
        if isinstance(cell, (np.ndarray, list, tuple, set)):
            cell_iterable = cell
        else:
            cell_iterable = [cell]

        # Lowercase all items for comparison
        cell_set = set(str(item).lower() for item in cell_iterable)

        # Must contain all and only those keywords
        return cell_set == target_set

    mask = df[col].apply(matches_exact_keywords)
    return df[mask]


def filter_dataframe(df: pd.DataFrame, filter_columns = []) -> pd.DataFrame:
    """
    Adds a UI on top of a dataframe to let viewers filter columns.
    Widgets are arranged in rows of 3 columns.
    """
    modify = st.checkbox("Add filters", value = False)

    if not modify:
        return df

    df = df.copy()

    # Convert datetimes into a standard format
    for col in df.columns:
        if is_object_dtype(df[col]):
            try:
                df[col] = pd.to_datetime(df[col])
            except Exception:
                pass

        if is_datetime64_any_dtype(df[col]):
            df[col] = df[col].dt.tz_localize(None)

    modification_container = st.container()

    with modification_container:
        
        if not filter_columns:
            filter_columns = df.columns
                
        to_filter_columns = st.multiselect("Filter dataframe on", filter_columns)

        # Arrange widgets in rows of 3
        for i in range(0, len(to_filter_columns), 3):
            row_cols = st.columns(3)
            for j, column in enumerate(to_filter_columns[i:i+3]):
                col_widget = row_cols[j]

                # Handle list/array/dict columns with keyword filtering
                if df[column].apply(lambda x: isinstance(x, (np.ndarray, list, dict))).any():
                    
                    col_widget.write(column)

                    if f"keywords_{column}" not in st.session_state:
                        st.session_state[f"keywords_{column}"] = set()
                    if f"selected_keywords_{column}" not in st.session_state:
                        st.session_state[f"selected_keywords_{column}"] = []

                    col_widget.text_input(
                        f"Enter {column} keyword:",
                        key=f"new_keyword_{column}",
                        on_change=add_keyword_1,
                        args=(column,),
                    )

                    col_widget.multiselect(
                        "Current Keywords:",
                        options=sorted(st.session_state[f"keywords_{column}"]),
                        default=sorted(st.session_state[f"keywords_{column}"]),
                        key=f"keyword_multiselect_{column}",
                        on_change=on_multiselect_change,
                        args=(column,),
                    )

                    exact_match = col_widget.checkbox("Exact match",
                                            key=f"exact_match_{column}")
                    

                    curr_filter_list = st.session_state[f"keywords_{column}"]
                    
                    if exact_match:
                        df = filter_rows_by_exact_keywords(df, column, curr_filter_list)
                    else:
                        df = filter_rows_by_all_keywords(df, column, curr_filter_list)

                # Treat categorical columns
                elif is_categorical_dtype(df[column]) or df[column].nunique() < 10:
                    user_cat_input = col_widget.multiselect(
                        f"Values for {column}",
                        df[column].unique(),
                        default=list(df[column].unique()),
                    )
                    df = df[df[column].isin(user_cat_input)]

                # Numeric columns
                elif is_numeric_dtype(df[column]):
                    _min = float(df[column].min())
                    _max = float(df[column].max())
                    step = (_max - _min) / 100
                    user_num_input = col_widget.slider(
                        f"Values for {column}",
                        _min,
                        _max,
                        (_min, _max),
                        step=step,
                    )
                    df = df[df[column].between(*user_num_input)]

                # Datetime columns
                elif is_datetime64_any_dtype(df[column]):
                    user_date_input = col_widget.date_input(
                        f"Values for {column}",
                        value=(df[column].min(), df[column].max()),
                    )
                    if len(user_date_input) == 2:
                        start_date, end_date = map(pd.to_datetime, user_date_input)
                        df = df.loc[df[column].between(start_date, end_date)]

                # Fallback: text/regex filtering
                else:
                    user_text_input = col_widget.text_input(f"Substring or regex in {column}")
                    if user_text_input:
                        df = df[df[column].str.contains(user_text_input, na=False)]

    return df


def display_images(trimmed_df, n_display = 50):
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
                        **Listing date:** {to_str(row.get("listing_date", []))}  
                        **Price:** {to_str(row.get("price", []))}  
                        **Monthly sales:** {listing_date}
                        """
                    )

                except (FileNotFoundError, UnidentifiedImageError, OSError):
                    st.warning(f"⚠️ Could not load image: {img_path}")


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
            
    st.markdown("#### 竞争对手分析")

    # get cutoff qty for our asin
    if our_asin in asin_price_sales.ASIN.values:
        # st.write(f"{our_asin} is in asin_price_sales")
        our_asin_qty = asin_price_sales[asin_price_sales.ASIN == our_asin].monthly_sales.values[0]
    else:
        # st.write(f"{our_asin} is NOT in asin_price_sales")
        our_asin_qty = 0 

    cutoff_qty_input = st.text_input("最少平均月销量", value=int(our_asin_qty * SALES_CUTOFF_MARGIN), width = 150)

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
            st.session_state['df'][DISPLAY_COLS], 
            on='ASIN'
        )
    else:
        # return empty DataFrame with same columns
        rival_asins = pd.DataFrame(columns=['ASIN', 'monthly_sales', 'url', 'product_title', 'listing_date'])
            
    filter_columns = ['ASIN', 'brand', 'price'] 
    trimmed_df = filter_dataframe(rival_asins, filter_columns) 
    st.dataframe(trimmed_df)

    display_images(trimmed_df)

    st.markdown("#### 快速增长的")

    growth_cutoff = st.text_input("三月销量增长率(%)", value=int(GROWTH_CUTOFF*100), width = 150)

    try:
        growth_cutoff = float(growth_cutoff) / 100
    except ValueError:
        st.error("Please enter a valid integer for cutoff quantity.")
        growth_cutoff = GROWTH_CUTOFF

    fast_growing_asins = get_fast_growing_asins(sales, asin_price_sales, growth_cutoff = growth_cutoff, sales_cutoff = 200)
    fast_growing_asins = fast_growing_asins.merge(df[DISPLAY_COLS], on = 'ASIN')
    st.write(fast_growing_asins)


    
    display_images(fast_growing_asins)