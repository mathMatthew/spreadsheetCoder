import os
import json
from datetime import datetime
import pytest

# Import your internal modules
import transpile_python as tp
import conversion_rules as cr
import conv_tracker as ct

example_dir = "./examples"
output_dir = "./sc_output_files"  # Adjusted for simplicity

@pytest.mark.parametrize("xml_file, mode", [
    ("endDateDays.XML", "build"),
    ("endDateDays.XML", "complete"),
    ("endDateDays.XML", "supplement")
     "endDateDays.XML",
    "av_bal.XML",
    "test_power.XML",
    "CmplxPeriod.XML",
    "myPandL.XML",
    "ranch.XML",
    "sumemup.XML"
])
def test_transpilation(xml_file):
    start = datetime.now()
    print(f"Testing {xml_file} - Start Time: {start.strftime('%H:%M:%S')}")

    conversion_tracker = ct.initialize_conversion_tracker()
    overrides = {}
    mode = "build"

    code, conversion_rules = tp.transpile(
        xml_file, example_dir, mode, conversion_tracker, overrides
    )

    assert code, f"Transpilation failed for {xml_file}"

    base_file_name = os.path.splitext(xml_file)[0]
    output_file = os.path.join(output_dir, base_file_name + ".py")
    conversion_rules_file = os.path.join(output_dir, base_file_name + "_conversion_rules.json")
    conv_tracker_file = os.path.join(output_dir, base_file_name + "_conversion_tracker.json")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(output_file, "w") as f:
        f.write(code)

    # Assuming these functions return success/failure or True/False
    assert cr.serialize_and_save_rules(conversion_rules, conversion_rules_file), "Failed to write conversion rules"
    with open(conv_tracker_file, "w") as f:
        json.dump(conversion_tracker, f, indent=2)

    # TODO: Here, you should add any steps necessary to verify the transpilation was successful,
    # such as running the produced Python code and checking its output or behavior.
    # This might involve importing and executing the generated Python code and validating the results.

    print(f"Testing {xml_file} - Total Time: {datetime.now() - start}")
