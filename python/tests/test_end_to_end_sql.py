import os
from datetime import datetime
import pytest

# Import your internal modules
import transpile_sql as ts
import conversion_rules as cr
import conv_tracker as ct

@pytest.mark.parametrize(
    "xml_file, mode",
    [
        ("endDateDays.XML", "complete"),
        ("myPandL.XML", "complete"),
        ("ranch.XML", "supplement"), #note: cannot run ranch using complete as a i purposefully removed some functions to test supplement mode
        ("endDateDays.XML", "build"),
        ("endDateMonths.XML", "build"),
        ("ageAtDate.XML", "build"),
        ("av_bal.XML", "build"),
        ("CmplxPeriod.XML", "build"),
        ("myPandL.XML", "build"),
        ("ranch.XML", "build"),
    ],
)
def test_transpilation(xml_file, mode, tmp_path):
    xml_test_files_dir = "./examples"

    start = datetime.now()
    print(f"Testing {xml_file} mode {mode} - Start Time: {start.strftime('%H:%M:%S')}")

    conversion_tracker = ct.initialize_conversion_tracker()
    overrides = {}

    code, conversion_rules = ts.transpile(
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
