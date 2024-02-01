from datetime import datetime, timedelta

def add_days_to_datetime(date_obj, days):
    #convert to seconds to capture fractionaly days
    seconds_to_add = days * 24 * 60 * 60
    return date_obj + timedelta(seconds=seconds_to_add)

def subtract_days_from_datetime(date_obj, days):
    # convert to seconds to capture fractionaly days
    seconds_to_add = days * 24 * 60 * 60
    return date_obj + timedelta(seconds=seconds_to_add)