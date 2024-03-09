import os
from datetime import datetime
import pytest

# Import your internal modules
import conversion_rules as cr
import conv_tracker as ct
import transpile_python as tp

@pytest.mark.parametrize(
    "xml_file, mode",
    [
        ("CmplxPeriod.XML", "supplement"), #note cannot run CmplxPeriod using complete as I purposefully removed the * function to test supplement mode
        ("sumemup.XML", "supplement"),
        ("endDateDays.XML", "build"),
        ("endDateMonths.XML", "build"),
        ("ageAtDate.XML", "build"),
        ("av_bal.XML", "build"),
        ("CmplxPeriod.XML", "build"),
        ("myPandL.XML", "build"),
        ("ranch.XML", "build"),
        ("endDateDays.XML", "complete"),
        ("av_bal.XML", "complete"),
    ],
)
def test_transpilation(xml_file, mode, tmp_path):

    xml_test_files_dir = "./examples"

    start = datetime.now()
    print(f"Testing {xml_file} mode {mode} - Start Time: {start.strftime('%H:%M:%S')}")

    conversion_tracker = ct.initialize_conversion_tracker()
    overrides = {}

    code, conversion_rules = tp.transpile(
        xml_file, xml_test_files_dir, mode, conversion_tracker, overrides
    )

    assert code, f"Transpilation failed for {xml_file} in {mode} mode"

    conversion_rules_file = os.path.join(
        tmp_path,  "conversion_rules.json"
    )

    assert cr.serialize_and_save_rules(
        conversion_rules, conversion_rules_file
    ), "Failed to write conversion rules"
    
    print(f"Testing {xml_file} - Total Time: {datetime.now() - start}")
