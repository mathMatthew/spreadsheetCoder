import transpile_python as tp
import conv_tracker as ct
import os, json
from datetime import datetime

#save the time in a variable and print the time
print("Current Time =", datetime.now().strftime("%H:%M:%S"))
start = datetime.now()


example_dir = "./examples"
output_dir = "../../../sc_output_files"
#xml_file = "endDateDays.XML"
#xml_file = "av_bal.XML"
#xml_file = "test_power.XML"
#xml_file = "CmplxPeriod.XML"
#xml_file = "myPandL.XML"
xml_file = "ranch.XML"

conversion_tracker = ct.empty_conversion_tracker()
overrides = {}
# overrides = {"auto_add_signatures": False}

code, fn_sig_translation_dict = tp.transpile(
    xml_file, example_dir, conversion_tracker, overrides
)

if code:
    base_file_name = os.path.splitext(xml_file)[0]
    output_file = os.path.join(output_dir, base_file_name + ".py")
    fn_sig_trans_file = os.path.join(output_dir, base_file_name + "_func_sigs.json")
    conv_tracker_file = os.path.join(
        output_dir, base_file_name + "_conversion_tracker.json"
    )

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(output_file, "w") as f:
        f.write(code)
        print(f"Code written to {output_file}")

    # write function signatures
    with open(fn_sig_trans_file, "w") as f:
        json.dump(fn_sig_translation_dict, f, indent=2)
        print(f"Function signatures written to {fn_sig_trans_file}")

    # write conversion tracker
    with open(conv_tracker_file, "w") as f:
        json.dump(conversion_tracker, f, indent=2)
        print(f"Conversion tracker written to {conv_tracker_file}")

print("Total Time = ", datetime.now() - start)