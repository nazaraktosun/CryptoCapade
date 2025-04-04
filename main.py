# main.py
import streamlit as st
from config import DEFAULT_CRYPTO_SYMBOLS, DEFAULT_TIMEFRAME
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
        3. View the resulting chart(s) on the main page.
        """
    )

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
    data = fetcher.get_crypto_data(symbol=crypto, days=30)
    st.write(f"Displaying simulated 30-day data for **{crypto}**:")
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
