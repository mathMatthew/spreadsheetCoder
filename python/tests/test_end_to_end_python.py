import os
import json
from datetime import datetime
import pytest

# Import your internal modules
import transpile_python as tp
import conversion_rules as cr
import conv_tracker as ct


@pytest.mark.parametrize(
    "xml_file, mode",
    [
        ("endDateDays.XML", "build"),
        ("endDateDays.XML", "complete"),
        ("endDateMonths.XML", "build"),
        ("ageAtDate.XML", "build"),
        ("av_bal.XML", "build"),
        ("av_bal.XML", "complete"),
        ("CmplxPeriod.XML", "build"),
        ("CmplxPeriod.XML", "supplement"), #note cannot run CmplxPeriod using complete as I purposefully removed the * function to test supplement mode
        ("myPandL.XML", "build"),
        ("ranch.XML", "build"),
        ("sumemup.XML", "complete"),
    ],
)
def test_transpilation(xml_file, mode, tmp_path):
    xml_test_files_dir = "./examples"
    output_dir = tmp_path

    start = datetime.now()
    print(f"Testing {xml_file} - Start Time: {start.strftime('%H:%M:%S')}")

    conversion_tracker = ct.initialize_conversion_tracker()
    overrides = {}

    code, conversion_rules = tp.transpile(
        xml_file, xml_test_files_dir, mode, conversion_tracker, overrides
    )

    assert code, f"Transpilation failed for {xml_file} in {mode} mode"

    base_file_name = os.path.splitext(xml_file)[0]
    output_file = os.path.join(output_dir, base_file_name + ".py")
    conversion_rules_file = os.path.join(
        output_dir, base_file_name + "_conversion_rules.json"
    )
    conv_tracker_file = os.path.join(
        output_dir, base_file_name + "_conversion_tracker.json"
    )

    with open(output_file, "w") as f:
        f.write(code)

    assert cr.serialize_and_save_rules(
        conversion_rules, conversion_rules_file
    ), "Failed to write conversion rules"
    
    with open(conv_tracker_file, "w") as f:
        json.dump(conversion_tracker, f, indent=2)

    print(f"Testing {xml_file} - Total Time: {datetime.now() - start}")
