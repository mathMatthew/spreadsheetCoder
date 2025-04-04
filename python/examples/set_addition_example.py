import os
import pandas as pd
import sys
import duckdb

# Add the root directory to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from set_addition import establish_founding_data_sets, incremental_add, global_reserve_tolerance, to_duckdb, get_db_connection

conn = get_db_connection()

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

flattened_df.to_csv(export_flat_1, index=False)

incremental_data = pd.read_csv(incremental_add_file)

all_facts_df, hierarchy_df, atomic_facts_df, flattened_df = incremental_add(
    all_facts_data=all_facts_df, 
    hierarchy_data=hierarchy_df, 
    atomic_facts_data=atomic_facts_df, 
    incremental_data=incremental_data, 
    reserve_tolerance=global_reserve_tolerance
    )

all_facts_df.to_csv(export_all_facts, index=False)
hierarchy_df.to_csv(export_hierarchy, index=False)
atomic_facts_df.to_csv(export_atomic_facts, index=False)
flattened_df.to_csv(export_flat_2, index=False)

# List of output files to compare
output_files = [
    export_all_facts, 
    export_hierarchy, 
    export_atomic_facts, 
    export_flat_1, 
    export_flat_2
]

def compare_csv_duckdb(file1, file2, threshold=0.01):
    file1_df = pd.read_csv(file1)
    file2_df = pd.read_csv(file2)
    filename1 = os.path.basename(file1).replace('.csv', '')
    filename2 = os.path.basename(file2).replace('.csv', '')
    table1 = to_duckdb(file1_df, False)
    table2 = to_duckdb(file2_df, False)
    
    # Get all column names for table1
    table_columns_query = f"SELECT column_name FROM information_schema.columns WHERE table_name='{table1}'"
    table_columns = conn.execute(table_columns_query).fetchdf()['column_name'].to_list()

    # Identify key columns (all columns except 'value' if it exists)
    key_columns = [col for col in table_columns if col != 'value']
    key_columns_str = ', '.join(key_columns)

    # Check if 'value' column exists
    has_value_col = 'value' in table_columns

    inner_query = f"""
        SELECT a.*, b.*,
            CASE
                WHEN a.{key_columns[0]} IS NULL THEN 'missing_in_{filename1}'
                WHEN b.{key_columns[0]} IS NULL THEN 'missing_in_{filename2}'
                {f"WHEN abs(a.value - b.value) > {threshold} THEN 'difference'" if has_value_col else ""}
                ELSE 'match'
            END AS issue
        FROM {table1} AS a
        FULL OUTER JOIN {table2} AS b
        USING ({key_columns_str})
    """

    # Outer query filters out exact matches
    join_query = f"""
        SELECT * FROM ({inner_query}) AS subquery
        WHERE issue != 'match'
    """

    mismatched = conn.execute(join_query).fetchdf()
    
    if mismatched.empty:
        print(f'✅ {filename1} matches {filename2}')
    else:
        print(f'❌ {filename1} does NOT match {filename2}')
        print(mismatched)

# Compare each output file with its reference
for file in output_files:
    reference_file = file.replace('.csv', '_reference.csv')
    if os.path.exists(reference_file):
        compare_csv_duckdb(reference_file, file)
    else:
        print(f'⚠️ Reference file {reference_file} not found.')
