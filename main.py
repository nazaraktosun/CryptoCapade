# main.py
import streamlit as st

# Must be the first Streamlit command
st.set_page_config(page_title="CryptoCapade", layout="wide")

from controllers.app_controller import CryptoAppController
from controllers.prediction_controller import PredictionController


def main():
    # Mode selection in the sidebar
    st.sidebar.title("Select Mode")
    mode = st.sidebar.radio("Which analysis do you want to perform?", options=["Analysis", "Prediction"])

    if mode == "Analysis":
        app_controller = CryptoAppController()
        app_controller.run()
    elif mode == "Prediction":
        prediction_controller = PredictionController()
        prediction_controller.run()


if __name__ == "__main__":
    main()
