import streamlit as st
import pandas as pd
import numpy as np
import re
import math

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from datetime import datetime

# Cohort labels are Chinese and reach matplotlib via the scatter legend, but
# matplotlib's default font has no CJK glyphs and would draw them as tofu boxes.
# Fall through to the first available; DejaVu Sans is the stock last resort.
plt.rcParams['font.sans-serif'] = [
    'Microsoft YaHei', 'SimHei', 'Noto Sans CJK SC', 'DejaVu Sans',
] + plt.rcParams['font.sans-serif']
plt.rcParams['axes.unicode_minus'] = False  # keep minus signs with a CJK font


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
    if sales.empty or prices.empty or df.empty:
        return pd.DataFrame()

    # pd series
    total_sales = get_monthly_sales(sales)
    waps = weighted_avg_price(sales, prices)
    n_listings = count_listings_by_month_range(df)

    summary = pd.concat([total_sales, waps, n_listings], axis=1)
    summary = summary.reset_index()
    summary.columns = ['month','total_sales', 'wavg_price', 'n_listings']

    # summary['n_listings'] = summary['n_listings'].fillna(method='ffill')
    summary['n_listings'] = summary['n_listings'].ffill()
    summary = summary.dropna()

    # replace price outlier
    summary['wavg_price'] = summary['wavg_price'].mask(
        (summary['wavg_price'] < (summary['wavg_price'].quantile(0.25) - 5 * (summary['wavg_price'].quantile(0.75) - summary['wavg_price'].quantile(0.25)))) |
        (summary['wavg_price'] > (summary['wavg_price'].quantile(0.75) + 5 * (summary['wavg_price'].quantile(0.75) - summary['wavg_price'].quantile(0.25))))
    ).ffill()

    summary['total_sales_pct_change'] = summary['total_sales'].pct_change().round(2)
    summary['n_listings_pct_change'] = summary['n_listings'].pct_change().round(2)

    # pct changes
    summary['sales_growth_yoy'] = (
    summary['total_sales'] / summary['total_sales'].shift(12) - 1
    ).round(2)

    summary['n_listings_growth_yoy'] = (
        summary['n_listings'] / summary['n_listings'].shift(12) - 1
    ).round(2)

    summary['sales_per_listing'] = summary['total_sales'] / summary['n_listings']       

    return summary.sort_values(by = 'month').round(2)


# @st.cache_data
# def plot_sales_timeseries(
#     df,
#     asin_list=None,
#     my_asins=None,
#     asin_col='ASIN',
#     start_date='2024-01-01',
#     top_n=3
# ):
#     """
#     df: sales df in wide format (ASIN + month columns with numeric values)
#     asin_list: list of ASINs to plot (default = all)
#     my_asins: list of ASINs to highlight in different color
#     asin_col: name of the ASIN column
#     start_date: earliest date to display (default '2024-01-01')
#     top_n: number of top ASINs to highlight (default 3)
#     """
    
#     # Identify month columns: anything except ASIN col
#     month_cols = [c for c in df.columns if c != asin_col]
    
#     # Convert start_date to datetime
#     start_dt = pd.to_datetime(start_date)
    
#     # Filter month columns to only those >= start_date
#     month_cols_filtered = [
#         c for c in month_cols 
#         if pd.to_datetime(c) >= start_dt
#     ]
    
#     # If asin_list not supplied, use all ASINs
#     if asin_list is None:
#         asin_list = df[asin_col].tolist()
    
#     # Filter df by the ASIN list
#     sub = df[df[asin_col].isin(asin_list)].copy()
    
#     # Get the most recent month column (last in filtered list)
#     most_recent_month = month_cols_filtered[-1]
    
#     # Find top N ASINs by most recent sales
#     top_asins = (
#         sub.nlargest(top_n, most_recent_month)[asin_col].tolist()
#     )
    
#     # Melt to long format using only filtered month columns
#     long = sub.melt(
#         id_vars=[asin_col],
#         value_vars=month_cols_filtered,
#         var_name='month',
#         value_name='sales'
#     )
    
#     # Convert the month col to datetime for proper plotting
#     long['month'] = pd.to_datetime(long['month'])
    
#     # --- Build Matplotlib fig for Streamlit ---
#     fig, ax = plt.subplots(figsize=(12, 6))
    
#     # Plot each ASIN
#     for asin in asin_list:
#         asin_data = long[long[asin_col] == asin]
        
#         # Highlight top N ASINs in softer red with moderate lines
#         if asin in top_asins:
#             ax.plot(
#                 asin_data['month'],
#                 asin_data['sales'],
#                 color='#D9534F',  # Softer red color
#                 linewidth=2,
#                 alpha=0.8,
#                 label=f"{asin}"
#             )
#         else:
#             ax.plot(
#                 asin_data['month'],
#                 asin_data['sales'],
#                 color='gray',
#                 alpha=0.3,
#                 linewidth=1
#             )
    
#     ax.set_xlabel("Month")
#     ax.set_ylabel("Sales")
#     ax.set_title("ASIN Sales Over Time")
    
#     # Show legend only for top ASINs, positioned at top right
#     if top_asins:
#         ax.legend(loc='upper right')
    
#     ax.grid(True, alpha=0.3)
#     fig.tight_layout()
    
#     # Render via Streamlit
#     st.pyplot(fig)

@st.cache_data
def plot_sales_timeseries(
    df,
    my_asins=None,
    highlighted_asin=None,
    asin_col='ASIN',
    start_date='2024-01-01',
    top_n=3
):
    """
    df: sales df in wide format (ASIN + month columns with numeric values)
    my_asins: list of ASINs to highlight in red
    highlighted_asin: single ASIN (or list) to highlight in blue
    asin_col: name of the ASIN column
    start_date: earliest date to display (default '2024-01-01')
    top_n: number of top my_asins to highlight (default 3)
    """
    
    month_cols = [c for c in df.columns if c != asin_col]
    start_dt = pd.to_datetime(start_date)
    month_cols_filtered = [c for c in month_cols if pd.to_datetime(c) >= start_dt]
    most_recent_month = month_cols_filtered[-1]

    asin_list = df[asin_col].tolist()

    # Normalize my_asins
    if my_asins is None:
        my_asins = []
    elif isinstance(my_asins, str):
        my_asins = [my_asins]

    # Normalize highlighted_asin
    if highlighted_asin is None:
        highlighted_asin = []
    elif isinstance(highlighted_asin, str):
        highlighted_asin = [highlighted_asin]
    highlighted_asin_set = set(highlighted_asin)

    # Find top N my_asins by most recent sales
    top_asins = []
    if my_asins:
        my_asins_df = df[df[asin_col].isin(my_asins)]
        top_asins = my_asins_df.nlargest(top_n, most_recent_month)[asin_col].tolist()

    long = df.melt(
        id_vars=[asin_col],
        value_vars=month_cols_filtered,
        var_name='month',
        value_name='sales'
    )
    long['month'] = pd.to_datetime(long['month'])

    fig, ax = plt.subplots(figsize=(12, 6))

    # Draw grey lines first, then blue, then red (so red is always on top)
    for asin in asin_list:
        if asin not in top_asins and asin not in highlighted_asin_set:
            asin_data = long[long[asin_col] == asin]
            ax.plot(asin_data['month'], asin_data['sales'],
                    color='gray', alpha=0.3, linewidth=1)

    for asin in highlighted_asin:
        asin_data = long[long[asin_col] == asin]
        ax.plot(asin_data['month'], asin_data['sales'],
                color='#5B8ED6', linewidth=2, alpha=0.8, label=f"{asin}")

    for asin in top_asins:
        asin_data = long[long[asin_col] == asin]
        ax.plot(asin_data['month'], asin_data['sales'],
                color='#D9534F', linewidth=2, alpha=0.8, label=f"{asin}")

    ax.set_xlabel("Month")
    ax.set_ylabel("Sales")
    ax.set_title("ASIN Sales Over Time")

    if top_asins or highlighted_asin:
        ax.legend(loc='upper right')

    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    st.pyplot(fig)

# @st.cache_data
# def scatter_price_vs_sales(prices_df, sales_df, n_months, asin_col='ASIN', our_asins=None, top_n=3):
#     """
#     Scatter plot of price vs total sales for the last n_months.

#     - Total sales are summed over the last n_months.
#     - Price is taken from the most recent month.
#     - our_asins (if supplied) is a list of ASINs to highlight in red.
#     - top_n: number of highest-selling highlighted ASINs to annotate (default 3)
#     Each point represents an ASIN.
#     """
#     # Identify month columns sorted ascending
#     month_cols = [c for c in prices_df.columns if c != asin_col]
#     month_cols = sorted(month_cols)
    
#     if n_months > len(month_cols):
#         raise ValueError(f"n_months={n_months} is larger than available months={len(month_cols)}")
    
#     # Last n_months
#     months_to_sum = month_cols[-n_months:]
#     most_recent_month = month_cols[-1]

#     # Check all months exist in both dfs
#     for df_name, df in [('prices_df', prices_df), ('sales_df', sales_df)]:
#         missing = [m for m in months_to_sum if m not in df.columns]
#         if missing:
#             raise ValueError(f"{df_name} is missing months: {missing}")

#     # Sum sales over last n_months
#     sales_sum = sales_df[[asin_col] + months_to_sum].copy()
#     sales_sum['total_sales'] = sales_sum[months_to_sum].sum(axis=1)

#     # Price from most recent month
#     prices_end = prices_df[[asin_col, most_recent_month]].copy()
#     prices_end = prices_end.rename(columns={most_recent_month: 'price'})

#     # Merge
#     merged = pd.merge(prices_end, sales_sum[[asin_col, 'total_sales']], on=asin_col)

#     # Drop rows with missing or zero values
#     merged = merged.dropna(subset=['price', 'total_sales'])
#     merged = merged[(merged['price'] != 0) & (merged['total_sales'] != 0)]
#     merged['monthly_sales'] = merged['total_sales']/n_months
    
#     # --- Plot for Streamlit ---
#     fig, ax = plt.subplots(figsize=(8, 6))

#     # Convert our_asins to list if needed, handle None
#     if our_asins is None:
#         our_asins = []
#     elif isinstance(our_asins, str):
#         our_asins = [our_asins]
    
#     # Plot all ASINs not in our_asins in blue
#     others = merged[~merged[asin_col].isin(our_asins)]
#     ax.scatter(others['price'], others['monthly_sales'], alpha=0.7, label='Other ASINs')

#     # Highlight our_asins in red
#     if our_asins:
#         ours = merged[merged[asin_col].isin(our_asins)]
#         if not ours.empty:
#             ax.scatter(ours['price'], ours['monthly_sales'], color='red', 
#                       label=f"Our ASINs ({len(ours)})")
            
#             # Get top N highest-selling highlighted ASINs
#             top_ours = ours.nlargest(top_n, 'monthly_sales')
            
#             # Annotate only the top N
#             for _, row in top_ours.iterrows():
#                 ax.annotate(
#                     f"{int(row['monthly_sales'])}",
#                     xy=(row['price'], row['monthly_sales']),
#                     xytext=(5, 5),  # offset 5 points right and up
#                     textcoords='offset points',
#                     fontsize=9,
#                     color='red',
#                     bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='red', alpha=0.7)
#                 )

#     ax.set_xlabel(f"Price ({most_recent_month})")
#     ax.set_ylabel(f"Total Sales (last {n_months} months)")
#     ax.set_title(f"Price vs Avg Monthly Sales — last {n_months} months")
#     ax.legend()
#     ax.grid(True)

#     # Render plot in Streamlit
#     st.pyplot(fig)

#     return merged.round(2)

@st.cache_data
def scatter_price_vs_sales(prices_df, sales_df, n_months, asin_col='ASIN', our_asins=None, top_n=3, 
                           cohort_asins=None, cohort_labels=None,
                           dot_transparency = 0.7):
    """
    Scatter plot of price vs total sales for the last n_months.

    - Total sales are summed over the last n_months.
    - Price is taken from the most recent month.
    - our_asins (if supplied) is a list of ASINs to highlight in red (takes priority over cohort colors).
    - top_n: number of highest-selling highlighted ASINs to annotate (default 3)
    - cohort_asins (if supplied) is a list of lists of ASINs; each sublist is colored distinctly.
    Each point represents an ASIN.
    """
    month_cols = [c for c in prices_df.columns if c != asin_col]
    month_cols = sorted(month_cols)
    
    if n_months > len(month_cols):
        raise ValueError(f"n_months={n_months} is larger than available months={len(month_cols)}")
    
    months_to_sum = month_cols[-n_months:]
    most_recent_month = month_cols[-1]

    for df_name, df in [('prices_df', prices_df), ('sales_df', sales_df)]:
        missing = [m for m in months_to_sum if m not in df.columns]
        if missing:
            raise ValueError(f"{df_name} is missing months: {missing}")

    sales_sum = sales_df[[asin_col] + months_to_sum].copy()
    sales_sum['total_sales'] = sales_sum[months_to_sum].sum(axis=1)

    prices_end = prices_df[[asin_col, most_recent_month]].copy()
    prices_end = prices_end.rename(columns={most_recent_month: 'price'})

    merged = pd.merge(prices_end, sales_sum[[asin_col, 'total_sales']], on=asin_col)
    merged = merged.dropna(subset=['price', 'total_sales'])
    merged = merged[(merged['price'] != 0) & (merged['total_sales'] != 0)]
    merged['monthly_sales'] = merged['total_sales'] / n_months
    
    fig, ax = plt.subplots(figsize=(8, 6))

    # Normalize inputs
    if our_asins is None:
        our_asins = []
    elif isinstance(our_asins, str):
        our_asins = [our_asins]
    our_asins_set = set(our_asins)

    if cohort_asins is None:
        cohort_asins = []
    # red is reserved for our_asins; 4 colors so the 4-cohort split doesn't wrap
    cohort_colors = ['green', 'blue', 'orange', 'purple']
    all_cohort_asins = {asin for group in cohort_asins for asin in group}

    # "Other" = not in our_asins and not in any cohort
    highlighted_asins = our_asins_set | all_cohort_asins
    others = merged[~merged[asin_col].isin(highlighted_asins)]
    if not others.empty:
        ax.scatter(others['price'], others['monthly_sales'], alpha=dot_transparency, label='Other ASINs')

    # Plot each cohort
    cohort_labels = cohort_labels or [f"Cohort {i+1}" for i in range(len(cohort_asins))]
    for i, group in enumerate(cohort_asins):
        color = cohort_colors[i % len(cohort_colors)]
        cohort_df = merged[merged[asin_col].isin(group) & ~merged[asin_col].isin(our_asins_set)]
        if not cohort_df.empty:
            ax.scatter(cohort_df['price'], cohort_df['monthly_sales'],
                       color=color, alpha=dot_transparency, label=cohort_labels[i])
    # our_asins always drawn last in red, overriding any cohort membership
    if our_asins:
        ours = merged[merged[asin_col].isin(our_asins_set)]
        if not ours.empty:
            ax.scatter(ours['price'], ours['monthly_sales'], color='red',
                       label=f"Our ASINs ({len(ours)})", zorder=5)

            for _, row in ours.nlargest(top_n, 'monthly_sales').iterrows():
                ax.annotate(
                    f"{int(row['monthly_sales'])}",
                    xy=(row['price'], row['monthly_sales']),
                    xytext=(5, 5),
                    textcoords='offset points',
                    fontsize=9,
                    color='red',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='red', alpha=0.7)
                )

    ax.set_xlabel(f"Price ({most_recent_month})")
    ax.set_ylabel(f"Avg Monthly Sales (last {n_months} months)")
    ax.set_title(f"Price vs Avg Monthly Sales — last {n_months} months")
    ax.legend()
    ax.grid(True)

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
    if df.empty:
        return
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
    
    # Draw horizontal line if line_value is supplied
    if line_value is not None:
        ax1.axhline(y=line_value, color='green', linestyle='--', label=f'our price')

    # --- X-AXIS FORMATTING ---
    # Major ticks every 3 months (with labels)
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    
    # Minor ticks every month (for gridlines, no labels)
    ax1.xaxis.set_minor_locator(mdates.MonthLocator(interval=1))
    
    # Enable grid for both major and minor ticks
    ax1.grid(True, which='major', alpha=0.5, linestyle='-')
    ax1.grid(True, which='minor', alpha=0.4, linestyle=':')
    
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




def plot_price_histogram(df):

        prices = df['price'].dropna()

        fig, ax = plt.subplots(figsize=(8, 4.5))

        # histogram
        ax.hist(
            prices,
            bins=30,
            edgecolor="white",
            linewidth=1.2,
            alpha=0.85
        )

        # titles
        ax.set_title("Price Distribution", fontsize=16, pad=12, weight="bold")
        ax.set_xlabel("Price ($)", fontsize=12)
        ax.set_ylabel("Number of Products", fontsize=12)

        # gridlines (the biggest visual improvement)
        ax.grid(
            axis="y",
            linestyle="--",
            linewidth=0.7,
            alpha=0.6
        )

        # remove ugly top/right borders
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # soften remaining borders
        ax.spines["left"].set_alpha(0.3)
        ax.spines["bottom"].set_alpha(0.3)

        # nicer tick formatting
        ax.tick_params(axis='both', labelsize=10)

        # add padding
        plt.tight_layout()

        st.pyplot(fig, width = 1200)


# --- Market share / concentration -------------------------------------------

COHORT_UNKNOWN = '未知'
BRAND_UNKNOWN = '未知'

# Cohorts split the most recent years out individually and collapse the rest.
# Once this month of the anchor year is reached, the current year has enough
# data to stand on its own and the oldest individual year is folded into the
# trailing bucket — 3 cohorts from July onwards, 4 before it.
COHORT_SPLIT_MONTH = 7

# rank 1 -> 5, anchored on the accent blue used elsewhere in the app.
# Kept light enough that the dark in-segment label stays legible on rank 1.
SHARE_PALETTE = ['#6E9EDD', '#8CB1E4', '#A8C3EB', '#C3D5F2', '#DDE7F9']
SHARE_OTHER_COLOR = '#9CA3AF'
SHARE_OTHER_FILL = '#E5E7EB'

# a segment narrower than this can't fit readable text
IN_BAR_LABEL_MIN_SHARE = 7.0


def get_recent_month_cols(sales_df, n_months=3, asin_col='ASIN'):
    """The last `n_months` month columns, oldest first.

    Unlike scatter_price_vs_sales this does not raise when the file has fewer
    months than requested — it just returns everything available.
    """
    month_cols = sorted([c for c in sales_df.columns if c != asin_col])
    if not month_cols:
        return []
    return month_cols[-min(n_months, len(month_cols)):]


def get_recent_sales(sales_df, n_months=3, asin_col='ASIN'):
    """Per-ASIN units sold over the last n_months.

    Returns (DataFrame[ASIN, recent_sales], months_used).
    """
    months = get_recent_month_cols(sales_df, n_months, asin_col)
    if not months:
        return pd.DataFrame(columns=[asin_col, 'recent_sales']), []

    recent = sales_df[[asin_col] + months].copy()
    recent['recent_sales'] = recent[months].sum(axis=1)
    # defensive: one row per ASIN even if the sheet repeats one
    recent = recent.groupby(asin_col, as_index=False)['recent_sales'].sum()

    return recent, months


def backfill_prices(prices_df, asin_col='ASIN'):
    """Fill price gaps per ASIN so revenue isn't understated by missing prices.

    0 and NaN both count as missing. Across the month columns, ascending:
      - interior gap (known on both sides) -> mean of the two bracketing values,
        so 5, na, na, 10 becomes 5, 7.5, 7.5, 10 (every gap month gets the same
        midpoint — deliberately not a linear interpolation)
      - trailing gap -> last known value
      - leading gap  -> first known value
      - no price in any month -> left at 0, contributing no revenue

    Sales are never backfilled — only prices.
    """
    months = sorted([c for c in prices_df.columns if c != asin_col])
    if not months:
        return prices_df.copy()

    out = prices_df.copy()
    known = out[months].mask(out[months] <= 0)

    prev = known.ffill(axis=1)   # last known value at or before each month
    nxt = known.bfill(axis=1)    # next known value at or after each month

    filled = (
        known
        .fillna((prev + nxt) / 2)  # interior gaps: bracketing midpoint
        .fillna(prev)              # trailing gaps: carry the last known forward
        .fillna(nxt)               # leading gaps: carry the first known back
        .fillna(0.0)               # ASIN with no price at all
    )
    out[months] = filled

    return out


def get_recent_revenue(sales_df, prices_df, n_months=3, asin_col='ASIN'):
    """Per-ASIN revenue over the last n_months: sum of units[m] * price[m].

    Prices are gap-filled by backfill_prices first, so a month with units sold
    but no recorded price still contributes. Prices here are listing prices
    (the per-unit qty division in load_data is commented out), so this is GMV
    per listing.

    Returns (DataFrame[ASIN, recent_revenue], months_used).
    """
    months = get_recent_month_cols(sales_df, n_months, asin_col)
    months = [m for m in months if m in prices_df.columns]
    if not months:
        return pd.DataFrame(columns=[asin_col, 'recent_revenue']), []

    units = sales_df[[asin_col] + months].groupby(asin_col, as_index=False).sum()
    # backfill across ALL months, then slice — a gap must see its neighbours
    price = backfill_prices(prices_df, asin_col)
    price = price[[asin_col] + months].drop_duplicates(subset=asin_col)

    # merge on ASIN — the two sheets are not guaranteed to be row-aligned
    merged = units.merge(price, on=asin_col, how='left', suffixes=('_units', '_price'))

    revenue = 0
    for m in months:
        revenue = revenue + merged[f'{m}_units'] * merged[f'{m}_price'].fillna(0)

    return merged[[asin_col]].assign(recent_revenue=revenue), months


def get_monthly_revenue(sales_df, prices_df, asin_col='ASIN'):
    """Total revenue per month across all ASINs.

    The revenue analogue of get_monthly_sales — returns a Series indexed by
    month-start datetime, so it drops straight into yoy_period_growth.
    """
    months = sorted(
        set(c for c in sales_df.columns if c != asin_col)
        & set(c for c in prices_df.columns if c != asin_col)
    )
    if not months:
        return pd.Series(dtype=float, name='total_revenue')

    units = sales_df[[asin_col] + months].groupby(asin_col, as_index=False).sum()
    price = backfill_prices(prices_df, asin_col)
    price = price[[asin_col] + months].drop_duplicates(subset=asin_col)

    merged = units.merge(price, on=asin_col, how='left', suffixes=('_units', '_price'))

    revenue = {
        m: float((merged[f'{m}_units'] * merged[f'{m}_price'].fillna(0)).sum())
        for m in months
    }

    out = pd.Series(revenue, name='total_revenue')
    out.index = pd.to_datetime(out.index)

    return out.sort_index()


# (divisor, suffix, decimals) largest first
_USD_UNITS = ((1e9, 'B', 2), (1e6, 'M', 2), (1e3, 'k', 1))


def format_usd_compact(value):
    """Format a dollar amount compactly: $2.27M, $226.8k, $950."""
    try:
        value = float(value)
    except (TypeError, ValueError):
        value = 0.0
    if value != value:  # NaN
        value = 0.0

    sign = '-' if value < 0 else ''
    magnitude = abs(value)

    for divisor, suffix, decimals in _USD_UNITS:
        # promote just below the divisor so 999,999 reads $1.00M, not $1000.0k
        if magnitude >= divisor * 0.9995:
            return f"{sign}${magnitude / divisor:.{decimals}f}{suffix}"

    return f"{sign}${magnitude:,.0f}"


def yoy_period_growth(monthly, n_months):
    """Year-over-year growth for the last `n_months` of a monthly series.

    `monthly` is a Series indexed by month-start datetimes, ascending — i.e.
    what get_monthly_sales returns. The prior window is looked up by *date*
    (each month minus 12), not by position, so a workbook with a missing month
    returns None rather than silently comparing against the wrong month.

    Returns None if the comparison isn't possible, else a dict with
    current / prior / growth / window / prior_window. `growth` is None when the
    prior window sums to zero — the raw numbers are still worth showing.
    """
    if monthly is None or len(monthly) < n_months:
        return None

    current_months = list(monthly.index[-n_months:])
    prior_months = [m - pd.DateOffset(months=12) for m in current_months]
    if any(m not in monthly.index for m in prior_months):
        return None

    current = float(monthly.loc[current_months].sum())
    prior = float(monthly.loc[prior_months].sum())

    return {
        'current': current,
        'prior': prior,
        'growth': (current / prior - 1) if prior > 0 else None,
        'window': (current_months[0], current_months[-1]),
        'prior_window': (prior_months[0], prior_months[-1]),
    }


def format_growth_pct(growth):
    """Format a growth fraction: +18%, +4.2%, +2,699% (×28.0), or —."""
    if growth is None:
        return "—"

    pct = growth * 100
    # one decimal for small moves so they don't all round to +0%
    text = f"{pct:+,.1f}%" if abs(pct) < 10 else f"{pct:+,.0f}%"
    if growth >= 2.0:
        text += f" (×{growth + 1:.1f})"

    return text


def get_reference_month(sales_df, asin_col='ASIN'):
    """The month the cohort split is anchored to: the workbook's latest month.

    Anchoring on the data rather than the clock means the same file always
    produces the same cohorts, however long after export it is opened. Falls
    back to today only when the frame carries no month columns at all.
    """
    months = get_recent_month_cols(sales_df, 1, asin_col)
    if not months:
        return pd.Timestamp.today().normalize()

    return pd.to_datetime(months[-1])


def get_cohort_years(reference):
    """Years to split out individually, NEWEST FIRST.

    Two years once the anchor reaches COHORT_SPLIT_MONTH (the current year has
    half a year of data behind it), three before that — so an early-in-the-year
    anchor keeps one extra year visible instead of burying it in the remainder.
    """
    ref = pd.Timestamp(reference)
    n_years = 2 if ref.month >= COHORT_SPLIT_MONTH else 3

    return [ref.year - i for i in range(n_years)]


def get_cohort_labels(reference):
    """Cohort labels for `reference`, OLDEST FIRST — so labels[-1] is newest.

    e.g. anchor 2026-07 -> ['2024及以前', '2025', '2026及以后']
         anchor 2026-02 -> ['2023及以前', '2024', '2025', '2026及以后']
    """
    years = get_cohort_years(reference)  # newest first

    labels = [f"{years[0]}及以后"] + [str(y) for y in years[1:]]
    labels.append(f"{years[-1] - 1}及以前")

    return labels[::-1]


def assign_cohort(listing_dates, reference):
    """Bucket listing dates into get_cohort_labels(reference); NaT -> COHORT_UNKNOWN.

    `reference` is required on purpose: defaulting it to today would quietly
    reintroduce the clock-anchored drift this replaced.
    """
    dates = pd.to_datetime(listing_dates, errors='coerce')
    years = dates.dt.year

    labels = get_cohort_labels(reference)       # oldest first
    split_years = get_cohort_years(reference)   # newest first

    cohort = pd.Series(COHORT_UNKNOWN, index=dates.index, dtype=object)
    cohort[years < split_years[-1]] = labels[0]
    for offset, year in enumerate(reversed(split_years)):
        cohort[years == year] = labels[offset + 1]
    # listings dated past the anchor year still belong to the newest cohort
    cohort[years >= split_years[0]] = labels[-1]

    return cohort


def normalize_brands(brands):
    """Merge whitespace/case variants of the same brand.

    Returns (key_series, {key: display_name}) where the display name is the
    most common original spelling seen for that key.
    """
    trimmed = brands.fillna('').astype(str).str.strip()
    keys = trimmed.str.lower().replace('', BRAND_UNKNOWN)

    display = {}
    for key, group in trimmed.groupby(keys):
        spellings = group[group != '']
        display[key] = spellings.mode().iloc[0] if not spellings.empty else BRAND_UNKNOWN

    return keys, display


def top_n_shares(labels, values, top_n=5, other_label='其他'):
    """Rank `labels` by summed `values` and split into top N + remainder.

    Returns (top_df[label, value, share], other_dict_or_None, total).
    Shares are percentages of `total` and always sum to 100.
    """
    grouped = (
        pd.DataFrame({'label': labels, 'value': values})
        .groupby('label', as_index=False)['value'].sum()
        .sort_values('value', ascending=False)
    )

    total = float(grouped['value'].sum())
    if total <= 0:
        return grouped.head(0).assign(share=[]), None, 0.0

    grouped['share'] = grouped['value'] / total * 100
    top = grouped.head(top_n).reset_index(drop=True)
    rest = grouped.iloc[top_n:]

    other = None
    if not rest.empty and rest['value'].sum() > 0:
        other = {
            'label': other_label,
            'value': float(rest['value'].sum()),
            'share': float(rest['share'].sum()),
            'count': int(len(rest)),
        }

    return top, other, total


def _share_segment_html(share, color, text, muted=False):
    # the remainder segment should recede behind the ranked ones — flat light
    # grey, opaque so the dark label stays readable in either Streamlit theme
    background = SHARE_OTHER_FILL if muted else color

    label = ''
    if text and share >= IN_BAR_LABEL_MIN_SHARE:
        label = (
            "<span style='font-size:0.78rem;font-weight:600;color:#1F2937;"
            "white-space:nowrap;overflow:hidden;text-overflow:ellipsis;'>"
            f"{text}</span>"
        )

    return (
        f"<div style='flex:0 0 {share:.4f}%;max-width:{share:.4f}%;height:38px;"
        f"background:{background};display:flex;align-items:center;"
        "justify-content:center;overflow:hidden;'>"
        f"{label}</div>"
    )


def render_share_bar(segments, other=None):
    """Render a 100%-stacked share bar plus a legend underneath.

    `segments` is a list of (label, share, color). `other` is an optional dict
    with keys label/share (and optionally count) rendered as a muted grey
    remainder. Shares are percentages and are expected to sum to 100.
    """
    if not segments and other is None:
        return

    bars = [
        _share_segment_html(share, color, f"{share:.0f}%")
        for _, share, color in segments
    ]

    legend_items = [
        (label, share, color, False) for label, share, color in segments
    ]

    if other is not None:
        other_text = other['label']
        if other.get('count'):
            other_text = f"{other['label']} ({other['count']})"
        bars.append(
            _share_segment_html(
                other['share'],
                SHARE_OTHER_FILL,
                f"{other_text} {other['share']:.0f}%",
                muted=True,
            )
        )
        legend_items.append((other_text, other['share'], SHARE_OTHER_FILL, True))

    legend = []
    for label, share, color, muted in legend_items:
        dot_style = (
            f"width:10px;height:10px;border-radius:2px;background:{color};"
            "flex:0 0 auto;"
        )
        if muted:
            # near-white chip needs an outline to be visible on a light theme
            dot_style += "box-shadow:inset 0 0 0 1px rgba(107,114,128,0.35);"
        legend.append(
            "<span style='display:inline-flex;align-items:center;gap:6px;"
            "font-size:0.82rem;line-height:1.4;'>"
            f"<span style='{dot_style}'></span>"
            f"<span>{label}</span>"
            f"<span style='opacity:0.65;'>{share:.0f}%</span>"
            "</span>"
        )

    html = (
        "<div style='display:flex;width:100%;border-radius:6px;overflow:hidden;'>"
        + "".join(bars)
        + "</div>"
        "<div style='display:flex;flex-wrap:wrap;gap:8px 18px;margin:10px 0 4px 0;'>"
        + "".join(legend)
        + "</div>"
    )

    st.markdown(html, unsafe_allow_html=True)

