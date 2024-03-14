import os, json
from datetime import datetime

# internal
import transpile_python as tp
import conversion_rules as cr
import conv_tracker as ct

# save the time in a variable and print the time
print("Current Time =", datetime.now().strftime("%H:%M:%S"))
start = datetime.now()

example_dir = "./examples"
output_dir = "../../../sc_output_files"
#xml_file = "endDateDays.XML"
# xml_file = "endDateMonths.XML"
# xml_file = "ageAtDate.XML"
xml_file = "av_bal.XML"
# xml_file = "CmplxPeriod.XML"
# xml_file = "myPandL.XML"
# xml_file = "ranch.XML"  # super slow to run this.
#xml_file = "sumemup.XML" #this is a nice example becaues it shows how we might want to introduce a custom rule 
    #just for a specific situation that we may not want to introduce into the language conversion rules file. 
    #This uses a 'sumnum' function--which allows non numbers inputs to get added as parameters and get ignored. 
    #It's a hack. I would never allow this sumnum function in something i wanted to run in production. Just clean up the excel file to say what you actually want it to do.
    #But it shows how we introduce a custom rule.

conversion_tracker = ct.initialize_conversion_tracker()
overrides = {}
mode = "build"  #'options:  'build' 'complete' 'supplement'
# overrides = {"auto_add_signatures": False}

code, conversion_rules = tp.transpile(
    xml_file, example_dir, mode, conversion_tracker, overrides
)

if code:
    base_file_name = os.path.splitext(xml_file)[0]
    output_file = os.path.join(output_dir, base_file_name + ".py")
    conversion_rules_file = os.path.join(
        output_dir, base_file_name + "_cr_py.json"
    )
    conv_tracker_file = os.path.join(
        output_dir, base_file_name + "_ct_py.json"
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

