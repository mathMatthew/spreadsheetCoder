import transpile_sql as ts
import conversion_rules as cr
import conv_tracker as ct
import os, json
from datetime import datetime

# save the time in a variable and print the time
print("Current Time =", datetime.now().strftime("%H:%M:%S"))
start = datetime.now()

example_dir = "./examples"
output_dir = "../../../sc_output_files"
# xml_file = "av_bal.XML"
# xml_file = "test_power.XML"
# xml_file = "CmplxPeriod.XML"
# xml_file = "myPandL.XML"
xml_file = "ranch.XML"
# xml_file = 'PeriodDiff.XML'

conversion_tracker = ct.empty_conversion_tracker()
overrides = {}
mode = "complete"  #'options:  'build' 'complete' 'supplement'

code, conversion_rules = ts.transpile(
    xml_file, example_dir, conversion_tracker, mode, overrides
)

if code:
    base_file_name = os.path.splitext(xml_file)[0]
    output_file = os.path.join(output_dir, base_file_name + ".SQL")
    conversion_rules_file = os.path.join(
        output_dir, base_file_name + "_conversion_rules.json"
    )
    conv_tracker_file = os.path.join(
        output_dir, base_file_name + "_conversion_tracker.json"
    )

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(output_file, "w") as f:
        f.write(code)
        print(f"Code written to {output_file}")

    # write function signatures
    cr.serialize_and_save_rules(conversion_rules, conversion_rules_file)
    print(f"Conversion rules file written to {conversion_rules_file}")

    # write conversion tracker
    with open(conv_tracker_file, "w") as f:
        json.dump(conversion_tracker, f, indent=2)
        print(f"Conversion tracker written to {conv_tracker_file}")

print("Total Time = ", datetime.now() - start)
