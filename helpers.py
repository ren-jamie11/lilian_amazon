import streamlit as st
import pandas as pd
import numpy as np
import re
import math

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from datetime import datetime


def get_today_yyyymm():
    return datetime.today().strftime("%Y-%m")

# Load Time Series
def filter_date_cols(cols):
    date_cols = [c for c in cols if re.match(r"^\d{4}-\d{2}", c)]
    return sorted(date_cols)

def clean_month_cols(cols):
    return [c.replace("($)", "") for c in cols]

def get_asin_and_months(raw_time_series_df):
    date_col_names = raw_time_series_df.columns.tolist()
    
    columns = filter_date_cols(date_col_names)
    columns = ['ASIN'] + columns
    
    res = raw_time_series_df[columns]
    res.columns = clean_month_cols(res.columns)

    return res

# Clean Time Series
def str_to_int(s: str) -> int | None:
    """
    Convert a string like '123,456' to integer 123456.
    Returns None if input is not a string or cannot be converted.
    """
    if isinstance(s, str):
        try:
            return int(s.replace(',', ''))
        except ValueError:
            return None
    elif isinstance(s, (int, float)):
        return int(s)
    else:
        return None

def clean_time_series(df, dtype=int):
    """
    Clean a time series DataFrame:
    - Fill NaNs with 0
    - Convert string numbers (with commas) to int or float
    - First column is assumed to be identifier and is not converted
    """
    df = df.fillna(0)

    if dtype == int:
        for col in df.columns[1:]:
            df[col] = df[col].apply(str_to_int) 

    elif dtype == float:
        for col in df.columns[1:]:
            df[col] = df[col].apply(lambda x: float(str(x).replace(',', '')) if x not in [None, ''] else 0.0)

    return df


# Get listing date for each ASIN
def get_listing_months(df, asin_col='ASIN'):
    # identify month columns
    month_cols = [c for c in df.columns if c != asin_col]

    # compute first valid index per row
    listing_month = df[month_cols].apply(lambda row: row.first_valid_index(), axis=1)

    # build result df
    result = df[[asin_col]].copy()
    result['listing_date'] = listing_month
    
    return result


# Get qty from product title
def extract_qty(title: str) -> int:
    if not isinstance(title, str):
        return 0
    
    s = title.lower()

    # 1️⃣ "set of x" OR "x set"
    m = re.search(r'set of\s+(\d+)', s)
    if m:
        return int(m.group(1))

    m = re.search(r'(\d+)\s+set\b', s)
    if m:
        return int(m.group(1))

    # 2️⃣ "x pack" or "x-pack"
    m = re.search(r'(\d+)\s*[- ]?\s*pack\b', s)
    if m:
        return int(m.group(1))

    # 3️⃣ "x pc", "x piece", "x-piece", "x pieces"
    m = re.search(r'(\d+)\s*[- ]?\s*(pc|piece|pieces)\b', s)
    if m:
        return int(m.group(1))

    # 4️⃣ Default
    return 1


# Monthly stats
def get_n_listings_from_time_series(df, asin_col='ASIN'):
    """
    Count how many listings have non-zero values in each month
    and return the result as a Pandas Series indexed by month.
    """
    # Identify month columns (all except ASIN column)
    month_cols = [c for c in df.columns if c != asin_col]

    # Count non-zero values column-wise
    count_nonzero = df[month_cols].ne(0).sum()

    # Convert index to datetime
    count_nonzero.index = pd.to_datetime(count_nonzero.index)

    return count_nonzero.rename("n_listings")

def get_monthly_sales(df, asin_col='ASIN'):
    """
    Sum total sales for each month across all listings.
    Returns a Pandas Series indexed by month.
    """
    # Identify month columns (all except ASIN column)
    month_cols = [c for c in df.columns if c != asin_col]

    # Sum values column-wise
    monthly_sum = df[month_cols].sum()

    # Convert index to datetime
    monthly_sum.index = pd.to_datetime(monthly_sum.index)

    return monthly_sum.rename("total_sales")

def weighted_avg_price(sales_df, price_df):
    """
    Compute weighted average monthly price from sales and price time series.
    Weighted by monthly sales volume for each ASIN.

    Returns: a pandas Series indexed by month, where each value is the
    weighted average price for that month.
    """
    # Align by ASIN and month (drop non-month col)
    sales = sales_df.set_index(sales_df.columns[0])
    prices = price_df.set_index(price_df.columns[0])
    
    # Make sure columns match
    common_months = sales.columns.intersection(prices.columns)
    sales = sales[common_months]
    prices = prices[common_months]

    # Weighted avg = sum(price * sales) / sum(sales) for each month
    weighted = (prices * sales).sum(axis=0) / sales.sum(axis=0)

    weighted.index = pd.to_datetime(weighted.index)

    return weighted

def count_listings_by_month_range(
    df,
    asin_col="ASIN",
    date_col="listing_date",
    min_month=None,
    max_month=None
):
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df["listing_month"] = df[date_col].dt.to_period("M")

    # Count new listings per month
    monthly_new = (
        df.groupby("listing_month")[asin_col]
        .nunique()
        .sort_index()
    )

    # Determine range bounds
    start = pd.Period(min_month, "M") if min_month else monthly_new.index.min()
    end = pd.Period(max_month, "M") if max_month else monthly_new.index.max()

    # Full monthly index
    full_index = pd.period_range(start, end, freq="M")

    # Fill gaps with 0 new listings
    monthly_new = monthly_new.reindex(full_index, fill_value=0)

    # Cumulative total
    cumulative = monthly_new.cumsum()

    # Convert PeriodIndex → datetime
    cumulative.index = pd.to_datetime(cumulative.index.astype(str))

    return cumulative


def summarize_price_sales(sales, prices, df):
    # pd series
    total_sales = get_monthly_sales(sales)
    waps = weighted_avg_price(sales, prices)
    n_listings = count_listings_by_month_range(df)

    summary = pd.concat([total_sales, waps, n_listings], axis=1)
    summary = summary.reset_index()
    summary.columns = ['month','total_sales', 'wavg_price', 'n_listings']

    summary = summary.dropna()

    return summary.sort_values(by = 'month').round(2)


# Plots

@st.cache_data
def plot_sales_timeseries(
    df,
    asin_list=None,
    my_asin=None,
    asin_col='ASIN'
):
    """
    df: sales df in wide format (ASIN + month columns with numeric values)
    asin_list: list of ASINs to plot (default = all)
    my_asin: your ASIN to highlight in different color
    asin_col: name of the ASIN column
    """
    
    # Identify month columns: anything except ASIN col
    month_cols = [c for c in df.columns if c != asin_col]
    
    # If asin_list not supplied, use all ASINs
    if asin_list is None:
        asin_list = df[asin_col].tolist()
    
    # Filter df by the ASIN list
    sub = df[df[asin_col].isin(asin_list)].copy()
    
    # Melt to long format
    long = sub.melt(
        id_vars=[asin_col],
        value_vars=month_cols,
        var_name='month',
        value_name='sales'
    )
    
    # Convert the month col to datetime for proper plotting
    long['month'] = pd.to_datetime(long['month'])
    
    # --- Build Matplotlib fig for Streamlit ---
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot each ASIN
    for asin in asin_list:
        asin_data = long[long[asin_col] == asin]
        
        # Highlight my ASIN in orange
        if asin == my_asin:
            ax.plot(
                asin_data['month'],
                asin_data['sales'],
                linewidth=3,
                label=f"{asin} (me)"
            )
        else:
            ax.plot(
                asin_data['month'],
                asin_data['sales'],
                alpha=0.5
            )
    
    ax.set_xlabel("Month")
    ax.set_ylabel("Sales")
    ax.set_title("ASIN Sales Over Time")
    
    if my_asin in asin_list:
        ax.legend()
    
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    
    # Render via Streamlit
    st.pyplot(fig)

@st.cache_data
def scatter_price_vs_sales(prices_df, sales_df, n_months, asin_col='ASIN', our_asin=None):
    """
    Scatter plot of price vs total sales for the last n_months.

    - Total sales are summed over the last n_months.
    - Price is taken from the most recent month.
    - our_asin (if supplied) is highlighted in red.
    Each point represents an ASIN.
    """
    # Identify month columns sorted ascending
    month_cols = [c for c in prices_df.columns if c != asin_col]
    month_cols = sorted(month_cols)
    
    if n_months > len(month_cols):
        raise ValueError(f"n_months={n_months} is larger than available months={len(month_cols)}")
    
    # Last n_months
    months_to_sum = month_cols[-n_months:]
    most_recent_month = month_cols[-1]

    # Check all months exist in both dfs
    for df_name, df in [('prices_df', prices_df), ('sales_df', sales_df)]:
        missing = [m for m in months_to_sum if m not in df.columns]
        if missing:
            raise ValueError(f"{df_name} is missing months: {missing}")

    # Sum sales over last n_months
    sales_sum = sales_df[[asin_col] + months_to_sum].copy()
    sales_sum['total_sales'] = sales_sum[months_to_sum].sum(axis=1)

    # Price from most recent month
    prices_end = prices_df[[asin_col, most_recent_month]].copy()
    prices_end = prices_end.rename(columns={most_recent_month: 'price'})

    # Merge
    merged = pd.merge(prices_end, sales_sum[[asin_col, 'total_sales']], on=asin_col)

    # Drop rows with missing or zero values
    merged = merged.dropna(subset=['price', 'total_sales'])
    merged = merged[(merged['price'] != 0) & (merged['total_sales'] != 0)]
    merged['monthly_sales'] = merged['total_sales']/n_months
    
    # --- Plot for Streamlit ---
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot all ASINs in blue
    others = merged[merged[asin_col] != our_asin]
    ax.scatter(others['price'], others['monthly_sales'], alpha=0.7, label='Other ASINs')

    # Highlight our_asin in red if supplied
    if our_asin is not None and our_asin in merged[asin_col].values:
        ours = merged[merged[asin_col] == our_asin]
        ax.scatter(ours['price'], ours['monthly_sales'], color='red', label=f"Our ASIN: {our_asin}")

    ax.set_xlabel(f"Price ({most_recent_month})")
    ax.set_ylabel(f"Total Sales (last {n_months} months)")
    ax.set_title(f"Price vs Avg Monthly Sales — last {n_months} months")
    ax.legend()
    ax.grid(True)

    # Render plot in Streamlit
    st.pyplot(fig)

    return merged.round(2)


@st.cache_data
def plot_ts_two_cols(
    df,
    date_col,
    value_col1,
    value_col2,
    line_value=None,
    start_date=None,
    end_date=None
):
    # Ensure the date column is in datetime format
    df[date_col] = pd.to_datetime(df[date_col])
    
    if start_date is not None:
        df = df[df[date_col] >= start_date]

    if end_date is not None:
        df = df[df[date_col] <= end_date]

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot first value column on left y-axis
    ax1.plot(df[date_col], df[value_col1], marker='o', linestyle='-', color='blue', label=value_col1)
    ax1.set_xlabel(date_col)
    ax1.set_ylabel(value_col1, color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.grid(True, alpha=0.3)

    # Draw horizontal line if line_value is supplied
    if line_value is not None:
        ax1.axhline(y=line_value, color='green', linestyle='--', label=f'our price')

    # --- X-AXIS FORMATTING ---
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.xticks(rotation=45)

    # Plot second value column on right y-axis
    ax2 = ax1.twinx()
    ax2.plot(df[date_col], df[value_col2], marker='s', linestyle='--', color='red', label=value_col2)
    ax2.set_ylabel(value_col2, color='red')
    ax2.tick_params(axis='y', labelcolor='red')

    # Combine legends
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')

    plt.title(f'{value_col1} and {value_col2} over {date_col}')
    plt.tight_layout()
    
    st.pyplot(fig)




# Get rising competitors
@st.cache_data
def sales_pct_diff(df, n_months, asin_col='ASIN'):
    """
    Calculate percent difference in sales from most recent month to n_months ago for each ASIN.
    
    Parameters:
    - df: sales DataFrame in wide format (ASIN + month columns)
    - n_months: how many months back to compare
    - asin_col: name of ASIN column
    
    Returns:
    - DataFrame with ASIN and pct_diff
    """
    # Identify month columns sorted ascending
    month_cols = [c for c in df.columns if c != asin_col]
    month_cols = sorted(month_cols)
    
    # Make sure we have enough months
    if n_months >= len(month_cols):
        raise ValueError(f"n_months={n_months} is too large; only {len(month_cols)} months available.")
    
    recent_col = month_cols[-1]
    past_col = month_cols[-(n_months+1)]
    
    result = df[[asin_col, past_col, recent_col]].copy()
    
    # Percent change formula: (recent - past) / past * 100
    result['pct_diff'] = (result[recent_col] - result[past_col]) / result[past_col]
    
    return result[[asin_col, 'pct_diff']].sort_values(by = "pct_diff", ascending = False)

@st.cache_data
def get_fast_growing_asins(sales, asin_price_sales, growth_cutoff = 0.5, sales_cutoff = 200):
    pct_growths = sales_pct_diff(sales, 3).round(2)
    fast_growers = pct_growths[pct_growths.pct_diff >= growth_cutoff]
    fast_growing_asin_list = fast_growers.ASIN.values.tolist()

    fast_growing_asins = asin_price_sales[asin_price_sales.ASIN.isin(fast_growing_asin_list)]
    fast_growing_asins = fast_growing_asins[fast_growing_asins.monthly_sales >= sales_cutoff]
    fast_growing_asins = fast_growing_asins.merge(pct_growths, on = 'ASIN', how = 'inner')

    return fast_growing_asins