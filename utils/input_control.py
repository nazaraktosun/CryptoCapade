# utils/input_control.py

class InputControl:
    """
    InputControl provides helper methods for validating and controlling user inputs.
    """

    @staticmethod
    def validate_crypto_selection(selection, valid_options):
        """
        Validate that the selected crypto is among valid options.
        :param selection: Selected crypto symbol.
        :param valid_options: List of valid crypto symbols.
        :return: Boolean indicating if valid.
        """
        return selection in valid_options

    @staticmethod
    def validate_moving_average_window(window: int) -> bool:
        """
        Validate that the moving average window is a positive integer.
        :param window: Moving average window.
        :return: Boolean indicating if valid.
        """
        return isinstance(window, int) and window > 0
