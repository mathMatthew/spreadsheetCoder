"""
Lightweight wrapper for convert_xml that converts a directory of XML
files in the spreadsheetCoder 3 schema format to the SC 5 schema
"""
from lxml import etree
import os, re, shutil, datetime
from xml.dom import minidom  # for prettify

# local imports
import convert_xml as cxml


def set_paths(conversion_type):
    base_path = "../../../OneDrive/Documents/myDocs/sc_v2_data"
    if conversion_type == "functions":
        return {
            "before_xsd_path": base_path + "sc_3_schema.xsd",
            "after_xsd_path": base_path + "sc_5_schema.xsd",
            "pre_process_dir": base_path + "xml_functions_s3",
            "post_process_dir": base_path + "xml_functions",
        }
    elif conversion_type == "transforms":
        return {
            "before_xsd_path": base_path + "sc_3_schema.xsd",
            "after_xsd_path": base_path + "sc_5_schema.xsd",
            "pre_process_dir": base_path + "xml_transforms_s3",
            "post_process_dir": base_path + "xml_transforms",
        }
    else:
        raise ValueError("Invalid conversion type")


def prettify(lxml_element) -> str:
    # for some reason i couldn't get the lxml built in pretty_print to work right.
    # so using minidom here instead

    # Serialize the XML
    xml_str: str = etree.tostring(lxml_element, xml_declaration=True, encoding="utf-8")  # type: ignore
    reparsed = minidom.parseString(xml_str)
    pretty_xml = reparsed.toprettyxml(indent="  ")

    # Regular expression to move the 'name' attribute to the front for readability
    pattern = re.compile(r'(\<CodeCalculation)([^>]*?)\s+name="([^"]+)"([^>]*>)')
    pretty_xml = pattern.sub(r'\1 name="\3"\2\4', pretty_xml, 1)

    return pretty_xml


def process_directory(paths):
    # Create a backup directory for the post-process directory based on today's date
    pre_process_dir = paths["pre_process_dir"]
    post_process_dir = paths["post_process_dir"]

    backup_dir = (
        post_process_dir + "_backup_" + datetime.datetime.now().strftime("%Y%m%d")
    )
    if not os.path.exists(backup_dir):
        if os.path.exists(post_process_dir):
            shutil.copytree(post_process_dir, backup_dir)
            print(f"Backup created at {backup_dir}")
        else:
            print(f"No existing post-process directory to backup.")
    else:
        print(f"Backup already exists at {backup_dir}")

    # Clear the post-process directory if it exists, or create it if it doesn't
    if os.path.exists(post_process_dir):
        shutil.rmtree(post_process_dir)
    os.makedirs(post_process_dir)

    # Iterate over each file in the pre-process directory
    for filename in os.listdir(pre_process_dir):
        # Check if the file is an XML file
        if filename.upper().endswith(".XML"):
            input_file = os.path.join(pre_process_dir, filename)
            output_file = os.path.join(post_process_dir, filename)

            # Assuming cxml.load_3_or_5_tree and prettify are defined elsewhere
            converted_xml = cxml.load_3_or_5_tree(
                input_file, paths["before_xsd_path"], paths["after_xsd_path"]
            )

            with open(output_file, "w") as file:
                file.write(prettify(converted_xml))

            print(f"Processed {filename}")


def main():
    for conversion_type in ["functions", "transforms"]:
        print(f"Processing for conversion type: {conversion_type}")
        paths = set_paths(conversion_type)
        process_directory(paths)
        print(f"Completed processing for {conversion_type}\n")


if __name__ == "__main__":
    main()
