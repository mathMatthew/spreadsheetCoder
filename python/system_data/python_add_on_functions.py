import pandas as pd
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

def sc_if(condition, if_true, if_false):
    if condition:
        return if_true
    else:
        return if_false

def sc_compare_elements(my_list, value):
    return [element == value for element in my_list]

def sc_filter(result_list, condition_list, default):
    filtered_results = []
    
    for result in result_list:
        # Check each condition
        for condition in condition_list:
            if condition(result):
                filtered_results.append(result)
                break
        else:
            # If no conditions are met, append the default value
            filtered_results.append(default)
            
    return filtered_results

def sc_first_true(result_list, bool_list, default):
    for result, is_true in zip(result_list, bool_list):
        if is_true:
            return result
    return default

def sc_first_true_pd(result_col, bool_col, default) :
    filtered = result_col[bool_col]
    return filtered.iloc[0] if not filtered.empty else default

def sc_eomonth(start_date, months):
    new_date = start_date + relativedelta(months=months)
    # Set the day to the last day of the new month
    end_of_month = new_date.replace(day=1) + relativedelta(months=1) - relativedelta(days=1)
    return end_of_month

def add_days_to_datetime(date_obj, days):
    # convert to seconds to capture fractionaly days
    seconds_to_add = days * 24 * 60 * 60
    return date_obj + timedelta(seconds=seconds_to_add)

def subtract_days_from_datetime(date_obj, days):
    # convert to seconds to capture fractionaly days
    seconds_to_add = days * 24 * 60 * 60
    return date_obj + timedelta(seconds=seconds_to_add)

def sc_lookup(value, lookup_column, result_column):  
    exact_match = result_column[lookup_column == value]
    if not exact_match.empty:
        return exact_match.iloc[0]

    if lookup_column.is_monotonic_increasing :
        # Find the largest value that is less than 'value'
        less_than_value = lookup_column[lookup_column < value]
        if not less_than_value.empty:
            # Return the corresponding value from result_column
            index_of_nearest_value = less_than_value.idxmax()
            return result_column[index_of_nearest_value]

    # If value not found return fist item from results column
    return result_column.iloc[0]

