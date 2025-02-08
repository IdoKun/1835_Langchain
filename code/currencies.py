# pylint: disable=missing-docstring

# TODO: add some currency rates
RATES = {# $DELETE_BEGIN
    "USDEUR": 0.85,
    "GBPEUR": 1.13,
    "CHFEUR": 0.86,
    "EURGBP": 0.885
    }# $DELETE_END}

def convert(amount, currency):
    """returns the converted amount in the given currency
    amount is a tuple like (100, "EUR")
    currency is a string
    """
    # $CHALLENGIFY_BEGIN
    # 100 EUR to USD IS 100 * 0.85
    for key, value in RATES.items():
        if amount[1] == key[0:3] and currency == "EUR":
            return round(value * amount[0])
        if  amount[1] == key[3:6] and currency == key[0:3]:
            return round(amount[0] / value)
    # $CHALLENGIFY_END
