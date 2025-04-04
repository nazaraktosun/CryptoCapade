# main.py
import streamlit as st
from datetime import date, timedelta
from config import DEFAULT_CRYPTO_SYMBOLS
from utils.data_fetcher import DataFetcher
from utils.analysis import CryptoAnalysis
from utils.visualization import CryptoVisualizer
from utils.input_control import InputControl


def main():
    st.set_page_config(page_title="Crypto Analysis App", layout="wide")

    # Sidebar: How to Use instructions and user controls
    st.sidebar.title("How to Use")
    st.sidebar.info(
        """
        1. Select a cryptocurrency from the dropdown.
        2. Choose the analysis type and parameters.
        3. Select your desired analysis date range.
        4. View the resulting chart(s) on the main page.
        """
    )
    # Sidebar date range input
    st.sidebar.subheader("Select Analysis Date Range")
    default_end = date.today()
    default_start = default_end - timedelta(days=30)
    date_range = st.sidebar.date_input(
        "Date Range",
        value=[default_start, default_end],
        max_value=date.today()  # Prevent selecting future dates
    )

    # Ensure date_range is a list with two elements
    if isinstance(date_range, tuple):
        date_range = list(date_range)

    if not (isinstance(date_range, list) and len(date_range) == 2):
        st.error("Please select a valid start and end date.")
        return

    start_date, end_date = date_range

    if start_date > end_date:
        st.error("Start date must be earlier than or equal to end date.")
        return

    if end_date > date.today():
        st.error("End date cannot be in the future.")
        return

    # Sidebar input controls
    crypto = st.sidebar.selectbox("Select Cryptocurrency", DEFAULT_CRYPTO_SYMBOLS)
    analysis_type = st.sidebar.selectbox("Select Analysis Type", ["Price Chart", "Moving Average"])


    if analysis_type == "Moving Average":
        window = st.sidebar.number_input("Moving Average Window (days)", min_value=1, max_value=30, value=7)
    else:
        window = None

    # Main content
    st.title("Crypto Analysis Dashboard")

    # Data fetching
    fetcher = DataFetcher()
    try:
        data = fetcher.get_crypto_data(symbol=crypto, start_date=start_date, end_date=end_date)
    except Exception as e:
        st.error(str(e))
        return

    st.write(f"Displaying data for **{crypto}** from **{start_date}** to **{end_date}**:")
    st.dataframe(data)

    # Analysis and Visualization
    analyzer = CryptoAnalysis(data)
    visualizer = CryptoVisualizer()

    if analysis_type == "Price Chart":
        st.subheader(f"{crypto} Price Chart")
        fig = visualizer.plot_price(data, crypto)
        st.pyplot(fig)
    elif analysis_type == "Moving Average":
        if not InputControl.validate_moving_average_window(window):
            st.error("Invalid moving average window. Please choose a positive integer.")
        else:
            moving_avg = analyzer.calculate_moving_average(window=window)
            st.subheader(f"{crypto} Price with {window}-Day Moving Average")
            fig = visualizer.plot_moving_average(data, moving_avg, crypto, window)
            st.pyplot(fig)


if __name__ == "__main__":
    main()
