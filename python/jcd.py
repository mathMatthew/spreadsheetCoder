"""
this module converts jcd to json and vice-versa
jcd is exactly the same as json but with CDATA
"""
import os, json

jcd_char_to_token = {
    "\\": "__BACKSLASH__",
    "\n": "__NEWLINE__",
    '"': "__DOUBLE_QUOTE__",
    "/": "__FORWARD_SLASH__",
    "\t": "__TAB__",
    "\b": "__BACKSPACE__",
    "\f": "__FORM_FEED__",
    "\r": "__CARRIAGE_RETURN__",
}

json_char_to_escape = {
    '\\': '\\\\', # Backslash
    '\n': '\\n',  # New line
    '"': '\\"',   # Double quote
    '/': '\\/',   # Forward slash 
    '\t': '\\t',   # Tab
    '\b': '\\b',  # Backspace
    '\f': '\\f',  # Form feed
    '\r': '\\r',  # Carriage return
}

cdata_info = {
    "start": {
        "token": "__CDATA_START__",
        "replacement": "<![CDATA[\n"
    },
    "end": {
        "token": "__CDATA_END__",
        "replacement": "\n]]>"
    } 
}

def invert_dict(input_dict):
    return {v: k for k, v in input_dict.items()}

jcd_token_to_char = invert_dict(jcd_char_to_token)

jcd_token_to_char[cdata_info["start"]["token"]] = cdata_info["start"]["replacement"]
jcd_token_to_char[cdata_info["end"]["token"]] = cdata_info["end"]["replacement"]


def save_jcd(data, filepath):
    """
    Converts a Python dictionary to a .jcd format (JSON with CDATA) and saves it to a file.
    Only converts strings containing specific characters (e.g., newline) to CDATA format.
    """

    def preprocess_value(value, tokens, cdata_dict, special_chars):
        """
        Recursively process a value, replacing special characters with tokens and wrapping with CDATA if necessary.
        Handles strings, lists, and dictionaries.
        """
        if isinstance(value, str):
            if any(char in value for char in special_chars):
                for char, token in tokens.items():
                    value = value.replace(char, token)
                return cdata_dict["start"]["token"]  + value + cdata_dict["end"]["token"]
            else:
                return value
        elif isinstance(value, dict):
            return {
                k: preprocess_value(v, tokens, cdata_info, special_chars) for k, v in value.items()
            }
        elif isinstance(value, list):
            return [preprocess_value(item, tokens, cdata_info, special_chars) for item in value]
        else:
            return value

    # Special characters that trigger CDATA conversion
    special_chars = ["\n"]

    # Preprocess the dictionary
    processed_data = preprocess_value(data, jcd_char_to_token, cdata_info, special_chars)

    # Convert dictionary to JSON string
    json_string = json.dumps(processed_data, indent=2)

    # Replace tokens with actual characters and CDATA markers
    for token, char in jcd_token_to_char.items():
        json_string = json_string.replace(token, char)

    # Save JSON string to file
    with open(filepath, "w") as file:
        file.write(json_string)

def load_jcd(filepath):
    """
    Parses a .jcd file (JSON with CDATA) and converts it to a dictionary.
    """
    with open(filepath, 'r') as file:
        jcd_string = file.read()

    # Process each CDATA section
    while True:
        start_index = jcd_string.find(cdata_info["start"]["replacement"])
        if start_index == -1:
            break  # No more CDATA sections
        end_index = jcd_string.find(cdata_info["end"]["replacement"], start_index)
        if end_index == -1:
            raise ValueError(f"CDATA section not properly closed for file: {filepath}")
        
        # Extract the CDATA section
        cdata_section = jcd_string[start_index + len(cdata_info["start"]["replacement"]):end_index]

        # Replace actual characters with escape seq.
        for char, esc_chars in json_char_to_escape.items():
            cdata_section = cdata_section.replace(char, esc_chars)

        # Replace the CDATA section in the original string
        jcd_string = jcd_string[:start_index] + cdata_section + jcd_string[end_index + len(cdata_info["end"]["replacement"]):]

    #"jcd_string" is now a jcd string without CDATA sections; therefore, it is a JSON string
    json_string = jcd_string
    # Convert the string to a dictionary
    try:
        return json.loads(json_string)
    except json.JSONDecodeError as e:
        save_and_raise_err(json_string, e)


def save_and_raise_err(bad_json, error):
    save_location = "./errors/bad_json.txt"

    # Save the code string to the file
    with open(save_location, "w") as file:
        file.write(bad_json)
    print(f"JCD to dicationary conversion failed. After converted JCD to JSON, JSON format was invalid. Bad JSON saved to {save_location}")
    # Re-raise the original error
    raise error from None

def json_to_same_jcd_file(json_file):
    # if you want to start a JCD file but want to start with a JSON
    # you'll need this once at the start.
    # after that the update_jcd_json.py can be used
    assert json_file.endswith(".json")
    jcd_file = json_file[:-5] + ".jcd"
    convert_json_to_jcd_file(json_file, jcd_file)

def convert_jcd_to_json_file(jcd_path, json_path):
    this_dict = load_jcd(jcd_path)
    with open(json_path, "w") as file:
        file.write(json.dumps(this_dict, indent=2))
    print(f"Converted {jcd_path} to {json_path}")

def convert_json_to_jcd_file(json_path, jcd_path):
    with open(json_path, "r") as file:
        this_dict = json.load(file)
    save_jcd(this_dict, jcd_path)
    print(f"Converted {json_path} to {jcd_path}")

def main():
    #occasionally i first create the json and want the jcd. one way to handle this
    #is to come here and run this function on the json.
    json_to_same_jcd_file("./system_data/sql_supported_functions.json")

if __name__ == "__main__":
    main()

