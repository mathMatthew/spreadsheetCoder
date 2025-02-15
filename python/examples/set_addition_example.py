import os
import pandas as pd
import sys

# Add the root directory to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from set_addition import establish_founding_data_sets, incremental_add, global_reserve_tolerance


data_directory_inputs = "examples/set_addition"
data_directory_outputs = "data/set_addition"
if not os.path.exists(data_directory_outputs):
    os.makedirs(data_directory_outputs)


founding_data_file = os.path.join(data_directory_inputs, "test_data.csv")
incremental_add_file = os.path.join(data_directory_inputs, "test_data_incremental_add.csv")
export_all_facts = os.path.join(data_directory_outputs, "all_facts.csv")
export_hierarchy = os.path.join(data_directory_outputs, "hierarchy.csv")
export_atomic_facts = os.path.join(data_directory_outputs, "atomic_facts.csv")
export_flat_1 = os.path.join(data_directory_outputs, "flattened.csv")
export_flat_2 = os.path.join(data_directory_outputs, "flattened_2.csv")

founding_data = pd.read_csv(founding_data_file)

all_facts_df, hierarchy_df, atomic_facts_df, flattened_df = establish_founding_data_sets(
    incoming_data=founding_data, 
    reserve_tolerance=global_reserve_tolerance
    )

# save first data set.
flattened_df.to_csv(export_flat_1, index=False)

incremental_data = pd.read_csv(incremental_add_file)

all_facts_df, hierarchy_df, atomic_facts_df, flattened_df = incremental_add(
    all_facts_data=all_facts_df, 
    hierarchy_data=hierarchy_df, 
    atomic_facts_data=atomic_facts_df, 
    incremental_data=incremental_data, 
    reserve_tolerance=global_reserve_tolerance
    )

# save to second data set -- and all_facts plus hierarchy
all_facts_df.to_csv(export_all_facts, index=False)
hierarchy_df.to_csv(export_hierarchy, index=False)
atomic_facts_df.to_csv(export_atomic_facts, index=False)
flattened_df.to_csv(export_flat_2, index=False)

