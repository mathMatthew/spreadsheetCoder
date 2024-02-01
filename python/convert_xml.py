"""This module contains functions for converting XML from the spreadsheetCoder 3 schema 
format to the SC 5 schema format.
"""

from lxml import etree
import os, re
from xml.dom import minidom
from collections import OrderedDict


DEFAULT_SCHEMA_3_PATH = "./system_data/sc_3_schema.xsd"
DEFAULT_SCHEMA_5_PATH = "./system_data/sc_5_schema.xsd"

_schema_cache = {}


def load_schema(schema_path):
    """Load and return an XML schema from a given path, with caching."""
    if schema_path not in _schema_cache:
        with open(schema_path, "rb") as schema_file:
            schema_root = etree.XML(schema_file.read())  # type: ignore
            _schema_cache[schema_path] = etree.XMLSchema(schema_root)
    return _schema_cache[schema_path]


def _strip_whitespace(element):
    """Recursively strip whitespace from an element's text and tail."""
    if element.text:
        element.text = element.text.strip()
    for sub_element in element:
        _strip_whitespace(sub_element)
        if sub_element.tail:
            sub_element.tail = sub_element.tail.strip()


def _remap_node_ids(root, node_id_remap):
    # Update NodeDependencies using the node_id_remap
    for node_dep in root.find("NodeDependencies"):
        parent_node_id = node_dep.get("ParentNodeId")
        child_node_id = node_dep.get("ChildNodeId")
        if parent_node_id in node_id_remap:
            node_dep.set("ParentNodeId", node_id_remap[parent_node_id])
        if child_node_id in node_id_remap:
            node_dep.set("ChildNodeId", node_id_remap[child_node_id])

    # Update NodeId in Outputs using the node_id_remap
    for output in root.find("Outputs"):
        output_node_id = output.get("NodeId")
        if output_node_id in node_id_remap:
            output.set("NodeId", node_id_remap[output_node_id])


def _fix_function_nodes(root):
    for function_node in root.find("FunctionNodes"):
        _rename_attribute(function_node, "Name", "function_name")
        function_node.attrib["function_name"] = function_node.attrib[
            "function_name"
        ].upper()
        _order_attributes(function_node, ["node_type", "node_id", "function_name"])


def _fix_constant_nodes(root):
    # Dictionary to map each unique constant (value, type) pair to its first NodeId
    first_node_id_for_constant = {}

    # Dictionary to map all NodeIds that need to be replaced
    node_id_remap = {}

    # List to keep track of ConstantNodes to be removed
    nodes_to_remove = []

    # Populate the dictionaries
    for constant_node in root.find("ConstantNodes"):
        _fix_type(constant_node)
        _rename_attribute(constant_node, "Type", "data_type")
        value = constant_node.get("Value")
        data_type = constant_node.get("data_type")
        node_id = constant_node.get("NodeId")

        constant_key = (value, data_type)
        if constant_key not in first_node_id_for_constant:
            first_node_id_for_constant[constant_key] = node_id
        else:
            # Map subsequent NodeIds to the first NodeId for the same constant
            node_id_remap[node_id] = first_node_id_for_constant[constant_key]
            nodes_to_remove.append(constant_node)
        _order_attributes(constant_node, ["node_type", "node_id", "value", "data_type"])

    # Remove the identified ConstantNodes
    for node in nodes_to_remove:
        root.find("ConstantNodes").remove(node)

    _remap_node_ids(root, node_id_remap)


def _fix_type(element):
    type_mapping = {
        "0": "Text",
        "1": "Number",
        "2": "Boolean",
        "3": "Date",
        "4": "Range",
        "5": "Any",
        "Array[Text]": "ARRAY[Text]",
        "Array[Number]": "ARRAY[Number]",
        "Array[Boolean]": "ARRAY[Boolean]",
        "Table_Column[Text]": "TABLE_COLUMN[Text]",
        "Table_Column[Number]": "TABLE_COLUMN[Number]",
        "Table_Column[Boolean]": "TABLE_COLUMN[Boolean]",
    }
    # yyyy add newtypes here.
    if "Type" in element.attrib:
        element.attrib["Type"] = type_mapping[element.attrib["Type"]]


def _fix_outputs(root):
    for output in root.find("Outputs"):
        _fix_type(output)
        _rename_attribute(output, "NodeId", "node_id")
        _rename_attribute(output, "Type", "data_type")
        _rename_attribute(output, "Name", "output_name")
        _rename_attribute(output, "Id", "output_order")
        output.attrib["output_order"] = str(int(output.attrib["output_order"]) - 1)
        _order_attributes(
            output, ["output_order", "output_name", "data_type", "node_id"]
        )


def _fix_named_nodes(root):
    results = root.find("NamedNodes")
    if results is None or len(results) == 0:  # to avoid lxml's stupid FutureWarning.
        return
    for named_node in results:
        _rename_attribute(named_node, "NodeId", "node_id")
        _rename_attribute(named_node, "Name", "node_name")
        name_level = named_node.get("NameLevel")
        if name_level == "10":
            named_node.attrib["node_name_type"] = "address"
        elif name_level == "100":
            named_node.attrib["node_name_type"] = "alias"
        else:
            save_xml_and_raise(root, f"NameLevel is not 10 or 100: {name_level}")
        del named_node.attrib["NameLevel"]
        _order_attributes(named_node, ["node_id", "nodename", "node_name_type"])


def _fix_dependencies(root):
    for dependency in root.find("NodeDependencies"):
        _rename_attribute(dependency, "ParentNodeId", "parent_node_id")
        _rename_attribute(dependency, "ChildNodeId", "child_node_id")
        _rename_attribute(dependency, "ParentPosition", "parent_position")
        dependency.attrib["parent_position"] = str(
            int(dependency.attrib["parent_position"]) - 1
        )
        _order_attributes(
            dependency, ["child_node_id", "parent_position", "parent_node_id"]
        )


def _fix_test_cases(root):
    results = root.find("TestCases")
    if results is None or len(results) == 0:
        return
    for test_case in results:
        for item in test_case:
            if item.tag == "InputValue":
                _fix_type(item)
                _rename_attribute(item, "Type", "data_type")
                item.tag = "input_value"
            if item.tag == "OutputValue":
                _fix_type(item)
                _rename_attribute(item, "Type", "data_type")
                item.tag = "output_value"
        test_case.tag = "test_case"


def _fix_inputs(root):
    # Dictionary to map each InputId to its first NodeId
    first_node_id_for_input = {}

    # Dictionary to map all NodeIds that need to be replaced
    node_id_remap = {}

    # Populate the dictionaries
    for input_dep in root.find("InputDependencies"):
        input_id = input_dep.get("InputId")
        node_id = input_dep.get("NodeId")

        if input_id not in first_node_id_for_input:
            first_node_id_for_input[input_id] = node_id
        else:
            # Map subsequent NodeIds to the first NodeId for the same InputId
            node_id_remap[node_id] = first_node_id_for_input[input_id]

    # Remove InputDependencies element
    root.remove(root.find("InputDependencies"))

    # Create InputNodes from Inputs
    input_nodes = etree.Element("InputNodes")  # type: ignore
    for input_elem in root.find("Inputs"):
        _fix_type(input_elem)
        _rename_attribute(input_elem, "Type", "data_type")
        input_id = input_elem.get("InputId")
        node_id = first_node_id_for_input[input_id]

        input_node = etree.SubElement(input_nodes, "InputNode")  # type: ignore
        input_node.set("nodeid", node_id)
        for attr in input_elem.keys():
            input_node.set(attr.lower(), input_elem.get(attr))
        input_node.attrib["inputid"] = str(int(input_node.attrib["inputid"]) - 1)
        _rename_attribute(input_node, "name", "input_name")
        _rename_attribute(input_node, "inputid", "input_order")
        _order_attributes(
            input_elem,
            ["node_type", "node_id", "input_name", "input_order", "data_type"],
        )

    for node_dependency in root.find("NodeDependencies"):
        parent_node_id = node_dependency.get("ParentNodeId")
        if parent_node_id in node_id_remap:
            node_dependency.set("ParentNodeId", node_id_remap[parent_node_id])

    # Replace Inputs with InputNodes
    root.remove(root.find("Inputs"))
    root.insert(root.index(root.find("FunctionNodes")), input_nodes)


def _rename_attribute(element, old_name, new_name):
    if old_name in element.attrib:
        element.attrib[new_name] = element.attrib[old_name]
        del element.attrib[old_name]


def _add_standardized_child_nodes(root, all_nodes, node_types, node_type):
    for child_node in root.find(node_types):
        new_node = etree.SubElement(all_nodes, "Node")  # type: ignore
        new_node.set("node_type", node_type)
        for attr in child_node.keys():
            new_node.set(attr.lower(), child_node.get(attr))
        _rename_attribute(new_node, "nodeid", "node_id")


def _order_attributes(element, first_attributes):
    # Process the specified element
    new_attrib = OrderedDict()

    # Add specified first attributes in order
    for attr in first_attributes:
        if attr in element.attrib:
            new_attrib[attr] = element.attrib[attr]

    # Add the remaining attributes
    for attr, value in element.attrib.items():
        if attr not in first_attributes:
            new_attrib[attr] = value

    # Replace the element's attributes
    element.attrib.clear()
    element.attrib.update(new_attrib)


def load_3_or_5_tree(
    input_file, schema_3_path=DEFAULT_SCHEMA_3_PATH, schema_5_path=DEFAULT_SCHEMA_5_PATH
):
    schema_5 = load_schema(schema_5_path)

    # Parse the XML file
    tree = etree.parse(input_file, None)
    root = tree.getroot()
    if not schema_5.validate(root):
        tree = convert_from_3_to_5(tree, schema_3_path, schema_5_path)

    return tree


def _fix_root(root):
    for element in root.xpath("//*[@HasMultipleChildren]"):
        element.attrib.pop("HasMultipleChildren")

    if "HasMultipleOutputs" in root.attrib:
        del root.attrib["HasMultipleOutputs"]
    if "Version" in root.attrib:
        del root.attrib["Version"]
    _rename_attribute(root, "Name", "name")
    # root.attrib["name"] = root.attrib["name"].upper()


from lxml import etree


def save_xml_and_raise(xml_root, value_error_text):
    # Convert the XML element to a string
    xml_str = prettify(xml_root)

    # Write the XML string to a file
    with open("errors/error.xml", "w") as file:
        file.write(xml_str)

    value_error_text += " Problematic xml saved to errors/error.xml."
    raise ValueError(value_error_text)


def convert_from_3_to_5(
    tree, schema_3_path=DEFAULT_SCHEMA_3_PATH, schema_5_path=DEFAULT_SCHEMA_5_PATH
):
    schema_3 = load_schema(schema_3_path)
    schema_5 = load_schema(schema_5_path)
    schema_5_filename = os.path.basename(schema_5_path)
    xml_after_xsd_path = os.path.join(
        "..", schema_5_filename
    )  # consider fixing this. this assumes that the schema file isin the parent directory compared to the current file.

    root = tree.getroot()
    is_valid = schema_3.validate(root)
    if not is_valid:
        error_messages = "Attempting to validate against schema 3. XML is not valid.\n"
        for error in schema_3.error_log:
            error_messages += f"{error.message} at line {error.line}\n"
        save_xml_and_raise(root, error_messages)

    _strip_whitespace(root)
    _fix_inputs(root)
    _fix_constant_nodes(root)
    _fix_function_nodes(root)
    _fix_root(root)

    # replace the three different nodetypes (InputNodes, ConstantNodes, FunctionNodes) with the same nodetype (Node)
    #    designating the node_type as a property
    all_nodes = etree.Element("Nodes")  # type: ignore
    _add_standardized_child_nodes(root, all_nodes, "InputNodes", "input")
    _add_standardized_child_nodes(root, all_nodes, "ConstantNodes", "constant")
    _add_standardized_child_nodes(root, all_nodes, "FunctionNodes", "function")

    root.insert(root.index(root.find("InputNodes")), all_nodes)
    root.remove(root.find("InputNodes"))
    root.remove(root.find("ConstantNodes"))
    root.remove(root.find("FunctionNodes"))

    _fix_outputs(root)
    _fix_dependencies(root)
    _fix_test_cases(root)
    _fix_named_nodes(root)

    # Define the namespace for xsi
    xsi = "http://www.w3.org/2001/XMLSchema-instance"
    root.nsmap["xsi"] = xsi

    # Add the schema location attribute
    root.set(f"{{{xsi}}}noNamespaceSchemaLocation", xml_after_xsd_path)

    if not schema_5.validate(root):
        save_xml_and_raise(
            root, "Attempting to validate against schema 5. XML is not valid."
        )

    return tree


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


def main():
    input_file = "ranch.XML"
    transform: bool = False
    working_directory = "../../../OneDrive/Documents/myDocs/sc_v2_data"
    xsd_file_3 = "sc_3_schema.xsd"
    xsd_file_5 = "sc_5_schema.xsd"
    transform_xsd_file_3 = "sc_3_schema.xsd"  # for now.
    transform_xsd_file_5 = "sc_5_schema.xsd"
    if transform:
        xsd_file_3 = transform_xsd_file_3
        xsd_file_5 = transform_xsd_file_5
    # generate output_file name from input_file name
    output_file = os.path.splitext(input_file)[0] + "_s5.xml"
    input_file = os.path.join(working_directory, input_file)
    output_file = os.path.join(working_directory, output_file)
    xsd_file_3 = os.path.join(working_directory, xsd_file_3)
    xsd_file_5 = os.path.join(working_directory, xsd_file_5)

    tree = load_3_or_5_tree(input_file, xsd_file_3, xsd_file_5)

    with open(output_file, "w") as file:
        file.write(prettify(tree))


if __name__ == "__main__":
    main()
