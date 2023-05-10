"""
****************************************************************************************************
  __               _   _           _       
 / _|             | | | |         | |      
| |_ _ __ ___  ___| |_| |__  _   _| |_ ___ 
|  _| '__/ _ \/ __| __| '_ \| | | | __/ _ \
| | | | | (_) \__ \ |_| |_) | |_| | ||  __/
|_| |_|  \___/|___/\__|_.__/ \__, |\__\___|
                              __/ |        
                             |___/         
Quickstart: Tasty Bytes - Snowpark 101 for Data Science
Script:       streamlit_app.py    
Create Date:  2023-05-12 
Author:       Marie Coolsaet
Copyright(c): 2023 Snowflake Inc. All rights reserved.
****************************************************************************************************
Description: 
    A Streamlit app for surfacing predicted shift sales for locations where truck drivers can park.
****************************************************************************************************
SUMMARY OF CHANGES
Date(yyyy-mm-dd)    Author              Comments
------------------- ------------------- ------------------------------------------------------------
2023-05-12          Marie Coolsaet           Initial Quickstart Release
****************************************************************************************************

"""

# Import Python packages
import streamlit as st
import plotly.express as px
import json

# Import Snowflake modules
from snowflake.snowpark import Session
import snowflake.snowpark.functions as F
from snowflake.snowpark import Window

# Set Streamlit page config
st.set_page_config(
    page_title="Streamlit App: Snowpark 101",
    page_icon=":truck:",
    layout="wide",
)

# Add header and a subheader
st.header("Predicted Shift Sales by Location")
st.subheader("Data-driven recommendations for food truck drivers.")


# Use st.cache_data to only rerun when the query changes or after 10 min.
@st.cache_resource(ttl=3600)
def init_connection():
    # Get account credentials from a json file
    with open("data_scientist_auth.json") as f:
        data = json.load(f)
        username = data["username"]
        password = data["password"]
        account = data["account"]

    # Specify connection parameters
    connection_parameters = {
        "account": account,
        "user": username,
        "password": password,
        "role": "accountadmin",
        "warehouse": "tasty_dsci_wh",
        "database": "frostbyte_tasty_bytes_dev",
        "schema": "analytics",
    }

    # Create Snowpark session
    return Session.builder.configs(connection_parameters).create()


# Connect to Snowflake
session = init_connection()


# Get a list of cities
@st.cache_data
def get_cities(_session):
    return (
        _session.table("frostbyte_tasty_bytes_dev.analytics.shift_sales_v")
        .select("city")
        .distinct()
        .sort("city")
        .to_pandas()
    )


cities = get_cities(session)

# Create input widgets for cities and shift
with st.container():
    col1, col2 = st.columns(2)
    with col1:
        # Drop down to select city
        city = st.selectbox("City:", cities)

    with col2:
        # Select AM/PM Shift
        shift = st.radio("Shift:", ("AM", "PM"), horizontal=True)


# Get predictions for city and shift time
@st.cache_data
def get_predictions(_session, city, shift):
    # Get data and filter by city and shift
    snowpark_df = _session.table(
        "frostbyte_tasty_bytes_dev.analytics.shift_sales"
    ).filter((F.col("shift") == shift) & (F.col("city") == city))

    # Get rolling average
    window_by_location_all_days = (
        Window.partition_by("location_id", "shift")
        .order_by("date")
        .rows_between(Window.UNBOUNDED_PRECEDING, Window.CURRENT_ROW - 1)
    )

    snowpark_df = snowpark_df.with_column(
        "avg_location_shift_sales",
        F.avg("shift_sales").over(window_by_location_all_days),
    )

    # Get tomorrow's date
    date_tomorrow = (
        snowpark_df.filter(F.col("shift_sales").is_null())
        .select(F.min("date"))
        .collect()[0][0]
    )

    # Filter to tomorrow's date
    snowpark_df = snowpark_df.filter(F.col("date") == date_tomorrow)

    # Impute
    snowpark_df = snowpark_df.fillna(value=0, subset=["avg_location_shift_sales"])

    # Encode
    snowpark_df = snowpark_df.with_column("shift", F.iff(F.col("shift") == "AM", 1, 0))

    # Define feature columns
    feature_cols = [
        "MONTH",
        "DAY_OF_WEEK",
        "LATITUDE",
        "LONGITUDE",
        "CITY_POPULATION",
        "AVG_LOCATION_SHIFT_SALES",
        "SHIFT",
    ]

    # Call the inference user-defined function
    snowpark_df = snowpark_df.select(
        "location_id",
        "latitude",
        "longitude",
        "avg_location_shift_sales",
        F.call_udf(
            "udf_linreg_predict_location_sales", [F.col(c) for c in feature_cols]
        ).alias("predicted_shift_sales"),
    )

    return snowpark_df.to_pandas()


# Update predictions and plot when the "Update" button is clicked
if st.button("Update"):
    # Get predictions
    predictions = get_predictions(session, city, shift)

    # Plot on a map
    predictions["PREDICTED_SHIFT_SALES"].clip(0, inplace=True)
    fig = px.scatter_mapbox(
        predictions,
        lat="LATITUDE",
        lon="LONGITUDE",
        hover_name="LOCATION_ID",
        size="PREDICTED_SHIFT_SALES",
        color="PREDICTED_SHIFT_SALES",
        zoom=8,
        height=800,
        width=1000,
    )
    fig.update_layout(mapbox_style="open-street-map")
    st.plotly_chart(fig, use_container_width=True)
