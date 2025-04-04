# utils/visualization.py
import matplotlib.pyplot as plt

class CryptoVisualizer:
    """
    CryptoVisualizer handles the creation of charts for crypto data.
    """

    def plot_price(self, data, symbol: str):
        """
        Plot the closing price chart.
        :param data: DataFrame with Date and Close columns.
        :param symbol: Cryptocurrency symbol.
        """
        plt.figure(figsize=(10, 5))
        plt.plot(data["Date"], data["Close"], marker="o", linestyle="-")
        plt.title(f"{symbol} Closing Prices")
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.grid(True)
        plt.tight_layout()
        return plt

    def plot_moving_average(self, data, moving_avg, symbol: str, window: int):
        """
        Plot the closing price along with the moving average.
        :param data: DataFrame with Date and Close columns.
        :param moving_avg: Series with moving average values.
        :param symbol: Cryptocurrency symbol.
        :param window: Window size for moving average.
        """
        plt.figure(figsize=(10, 5))
        plt.plot(data["Date"], data["Close"], marker="o", label="Close Price")
        plt.plot(data["Date"], moving_avg, marker="x", label=f"{window}-Day MA", color="orange")
        plt.title(f"{symbol} Price and {window}-Day Moving Average")
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        return plt
