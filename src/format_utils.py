from babel.numbers import format_currency

def format_price(value: float, locale: str = 'en_US') -> str:
    """
    Format a float value as a currency string according to the locale.
    
    Args:
        value (float): The price value to format.
        locale (str): The locale string (e.g., 'en_US').
        
    Returns:
        str: Locale-formatted currency string.
    """
    return format_currency(value, 'USD', locale=locale)

def format_rmse(rmse_value: float) -> str:
    """
    Format the RMSE value for user-friendly display.
    """
    formatted = format_price(rmse_value)
    return f"Root Mean Squared Error (RMSE): {formatted}"
