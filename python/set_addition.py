import duckdb
import pandas as pd
import networkx as nx
import numpy as np
import uuid
from itertools import product


# Open DuckDB connection as a global
conn = duckdb.connect(database=':memory:')  # Use ':memory:' for an in-memory database, or specify a file path


# Predefined dimension names for auto-feeding
predefined_dim_names = {
    "filterDim1": "Metric",
    "filterDim2": "Product",
    "rowsDim1": "Geography",
    "rowsDim2": "Time",
}

global_reserve_tolerance = 0.01

def add_dimension_names(dim_mapping_df, founding_df, auto_feed_dim_names):
    #adds dimension names either by asking the user for them (normal) or
    #assigning them pre-defined names (auto_feed_dim_names=True) which is used 
    #for testing.
    for idx, row in dim_mapping_df.iterrows():
        if auto_feed_dim_names:
            dimension_name = predefined_dim_names.get(row["base_name"])
        else:
            unique_members = (
                founding_df[[row["code_column"], row["description_column"]]]
                .drop_duplicates()
                .to_string(index=False)
            )
            print(
                f"Unique members for dimension '{row['base_name']}':\n{unique_members}\n"
            )
            dimension_name = input(
                f"Enter the name for dimension '{row['base_name']}': "
            )

        dim_mapping_df.at[idx, "dimension_name"] = dimension_name

    return dim_mapping_df


def add_first_hierarchy_level(incoming_df, mapping_df, hierarchy_df):
    for mapping_row in mapping_df[mapping_df["type"] == "filter"].itertuples():
        unique_df = incoming_df.drop_duplicates(
            subset=mapping_row.code_column, keep="first"
        )
        if len(unique_df) > 1:
            raise ValueError(
                f"More than one unique code found for filter dimension '{mapping_row.base_name}'."
            )
        single_row = unique_df.iloc[[0]]
        single_row = single_row[
            [mapping_row.code_column, mapping_row.description_column]
        ]
        single_row.columns = ["code", "description"]
        single_row["parent"] = f"Top_{mapping_row.dimension_name}"
        single_row["dimension"] = mapping_row.dimension_name
        hierarchy_df = pd.concat([hierarchy_df, single_row], ignore_index=True)
    for mapping_row in mapping_df[mapping_df["type"] == "row"].itertuples():
        unique_df = incoming_df.drop_duplicates(
            subset=mapping_row.code_column, keep="first"
        )
        rel_level1 = unique_df[unique_df[mapping_row.rel_level_column] == 1]
        rel_level1 = rel_level1[
            [mapping_row.code_column, mapping_row.description_column]
        ]
        rel_level1.columns = ["code", "description"]
        rel_level1["parent"] = f"Top_{mapping_row.dimension_name}"
        rel_level1["dimension"] = mapping_row.dimension_name
        hierarchy_df = pd.concat([hierarchy_df, rel_level1], ignore_index=True)
    return hierarchy_df


def validate_and_merge_new_facts(prior_all_facts_df, new_facts_df, tolerance):
    # Identify key columns (all columns except the 'value' column)
    key_columns = new_facts_df.columns.difference(["value"])

    # Merge the old and new data on the key columns to identify overlapping records
    merged_df = pd.merge(
        prior_all_facts_df,
        new_facts_df,
        on=key_columns.tolist(),
        suffixes=("_old", "_new"),
        how="inner",
    )

    # Check for discrepancies in the 'value' column with a tolerance factor
    conflict_df = merged_df[
        abs(merged_df["value_old"] - merged_df["value_new"]) > tolerance
    ]

    if not conflict_df.empty:
        # If discrepancies exist, save the conflicting records (both old and new) to a CSV
        conflict_df.to_csv("data/conflicts.csv", index=False)

        # Raise an error or warning with details about the conflicts
        raise ValueError(
            f"Data conflict detected! {len(conflict_df)} conflicting records found (tolerance: {tolerance}). "
            f"Conflicts have been saved to 'data/conflicts.csv'."
        )

    # If no conflicts, proceed with concatenation of new and old facts
    new_all_facts_df = pd.concat(
        [prior_all_facts_df, new_facts_df], ignore_index=True
    ).drop_duplicates()

    # Identify new facts not present in the old dataset
    new_unique_facts = set_subtract(
        new_facts_df, prior_all_facts_df, key_columns.tolist()
    )

    return new_all_facts_df, new_unique_facts


def incremental_assign_dimension_names(incoming_df, hierarchy_df, dim_mapping_df):
    #  map the generic column names of the incoming df to the actual dimension names
    #  based on matching members from hierarchy df.

    # validate that filters have only one value which already exists in hierarchy
    for _, row in dim_mapping_df[dim_mapping_df["type"] == "filter"].iterrows():
        unique_filter_values = incoming_df[row["code_column"]].unique()
        if len(unique_filter_values) != 1:
            print(
                f"Validation failed: Multiple values found for filter dimension '{row['code_column']}': {unique_filter_values}"
            )
            return False, None
        filter_value = unique_filter_values[0]
        hierarchy_row = hierarchy_df[hierarchy_df["code"] == filter_value]
        if hierarchy_row.empty:
            print(
                f"Validation failed: Filter dimension value '{filter_value}' does not exist in the hierarchy."
            )
            return False, None
        else:
            # map the arbitrary column names to the correct dimension name based on the matching code.
            dim_mapping_df.loc[
                dim_mapping_df["code_column"] == row["code_column"], "dimension_name"
            ] = hierarchy_row.iloc[0]["dimension"]
            dim_mapping_df.loc[
                dim_mapping_df["code_column"] == row["code_column"], "top_level"
            ] = hierarchy_row.iloc[0]["level"]

    # for each row dimension,
    #    validate that all top level members already exist in hierarchy
    for _, row in dim_mapping_df[dim_mapping_df["type"] == "row"].iterrows():
        top_level_members = incoming_df[
            incoming_df[f'{row["base_name"]}_rel_level'] == 1
        ][row["code_column"]].unique()
        levels = set()
        dimension_names = set()
        for member in top_level_members:
            hierarchy_row = hierarchy_df[hierarchy_df["code"] == member]
            if hierarchy_row.empty:
                print(
                    f"Validation failed: Row dimension top-level member '{member}' does not exist in the hierarchy."
                )
                return False, None
            dimension_names.add(hierarchy_row.iloc[0]["dimension"])
            levels.add(hierarchy_row.iloc[0]["level"])

        if len(dimension_names) > 1:
            print(
                f"Validation failed: Row dimension top-level members '{top_level_members}' belong to different dimensions."
            )
            return False, None

        if len(levels) > 1:
            print(
                f"Validation failed: Row dimension top-level members '{top_level_members}' belong to different levels."
            )
            return False, None
        
        # Update dim_mapping_df for the row dimension with validated dimension name and level
        dim_mapping_df.loc[
            dim_mapping_df["code_column"] == row["code_column"], "dimension_name"
        ] = hierarchy_row.iloc[0]["dimension"]
        dim_mapping_df.loc[
            dim_mapping_df["code_column"] == row["code_column"], "top_level"
        ] = hierarchy_row.iloc[0]["level"]

    return True, dim_mapping_df


def build_hierarchy_mappings(hierarchy_df, include_level_1_child_records):
    """
    Constructs a dictionary of DataFrames containing child-parent mappings for each unique dimension.

    Each DataFrame maps codes to themselves and to their parent, stored by dimension names as keys.

    Returns:
        dict: A dictionary where keys are dimension names, and values are DataFrames containing:
              - 'prior_member': The original dimension code (either self or parent).
              - 'new_member': The corresponding dimension code (either self or parent).
    """

    hierarchy_mappings = {}

    # Loop through each unique dimension in the hierarchy DataFrame
    for dimension in hierarchy_df["dimension"].unique():
        # Filter the hierarchy DataFrame for the current dimension
        dim_df = hierarchy_df[
            (hierarchy_df["dimension"] == dimension) & (hierarchy_df["level"] >= 2)
        ]

        if not dim_df.empty:
            # Create the self-referential mapping (code to code)
            child_df = pd.DataFrame(
                {
                    "prior_member": dim_df["code"],
                    "new_member": dim_df["code"],
                    "new_member_type": "child",
                    "level": dim_df["level"],
                }
            )

            # Create the parent-child mapping (code to parent)
            parent_df = pd.DataFrame(
                {
                    "prior_member": dim_df["code"],
                    "new_member": dim_df["parent"],
                    "new_member_type": "parent",
                    "level": dim_df["level"] - 1,
                }
            ).dropna(subset=["new_member"])

            if include_level_1_child_records:
                dim_df_level1 = hierarchy_df[
                    (hierarchy_df["dimension"] == dimension)
                    & (hierarchy_df["level"] == 1)
                ]
                leve_1_child_df = pd.DataFrame(
                    {
                        "prior_member": dim_df_level1["code"],
                        "new_member": dim_df_level1["code"],
                        "new_member_type": "child",
                        "level": dim_df_level1["level"],
                    }
                )
                child_df = pd.concat([child_df, leve_1_child_df])

            # Combine the self-referential and parent-child mappings
            dim_df = pd.concat([child_df, parent_df]).drop_duplicates()

        # Add the dataset for this dimension to the dictionary
        hierarchy_mappings[dimension] = dim_df

    return hierarchy_mappings


def required_rows(facts_df, hierarchy_mappings):
    # This function will build a list of required rows to avoid "partials"
    # This process will produce a set of required rows which is approximately 2^n larger

    required_rows_df = facts_df.copy()
    required_columns = list(hierarchy_mappings.keys())

    for dimension in hierarchy_mappings.keys():
        dim_df = hierarchy_mappings[dimension]

        if not dim_df.empty:
            # Merge on the prior_member from the incoming data and the dimension-specific dataframe
            required_rows_df = required_rows_df.merge(
                dim_df, left_on=dimension, right_on="prior_member", how="inner"
            )

            # Rename the current dimension column to avoid conflicts
            required_rows_df.rename(
                columns={dimension: f"{dimension}_prior"}, inplace=True
            )

            # Rename the new_parent column to the dimension name
            required_rows_df.rename(columns={"new_member": dimension}, inplace=True)

            # Select only the required columns for the next iteration
            required_rows_df = required_rows_df[required_columns]
            required_rows_df = required_rows_df.drop_duplicates()

    return required_rows_df


def set_subtract(set1, set2, columns):
    # Subtract set2 from set1 based on the specified columns
    set1_subset = set1[columns]
    set2_subset = set2[columns]

    # Find rows in set1 that are not in set2 based on the specified columns
    mask = ~set1_subset.apply(tuple, axis=1).isin(set2_subset.apply(tuple, axis=1))

    # Return the full set1 rows where the mask is True
    result = set1[mask]
    return result


def missing_rows_all_ancestors(all_facts_df, hierarchy_df):
    # Build the hierarchy mappings for each dimension.
    hierarchy_mappings = build_hierarchy_mappings(hierarchy_df, False)

    # Get the list of initial missing rows based on all facts.
    initial_required_rows = required_rows(all_facts_df, hierarchy_mappings)

    # Perform initial set subtraction to get the missing rows
    missing_rows = set_subtract(
        initial_required_rows, all_facts_df, hierarchy_mappings.keys()
    )

    # Initialize a DataFrame to hold all required rows based on the initial missing rows
    build_result = pd.DataFrame(columns=missing_rows.columns)

    iteration_count = 0  # Initialize iteration counter

    while not missing_rows.empty:
        build_result = pd.concat([build_result, missing_rows])
        new_required_rows = required_rows(build_result, hierarchy_mappings)
        missing_rows = set_subtract(
            new_required_rows, build_result, hierarchy_mappings.keys()
        )
        iteration_count += 1
        if iteration_count >= 100:
            raise RuntimeError("Reached 100 iterations. This may indicate an infinite loop.")

    missing_required_rows = build_result
    if not missing_required_rows.empty:
        missing_required_rows.to_csv("data/missing_rows.csv", index=False)
        all_facts_df = all_facts_df.drop(columns=["value"])
        all_facts_df = pd.concat([all_facts_df, missing_required_rows], axis=0, ignore_index=True)
        all_facts_df = all_facts_df.drop_duplicates()
        # sort so that child code is always before parent code
        all_facts_df = sort_child_to_parent(all_facts_df, hierarchy_df)
        all_facts_df.to_csv("data/required_new_query.csv", index=False)
        raise Exception(
            "Missing required rows. See 'missing_rows.csv' and 'required_new_query.csv'."
        )

    return True


def denormalize_balanced_dimension(hierarchy, dimension):
    # Filter hierarchy for the given dimension
    dimension_hierarchy = hierarchy[hierarchy["dimension"] == dimension].copy()

    # Start with the lowest level and move up to create the dernomalized structure
    max_level = dimension_hierarchy["level"].max()

    if max_level == 0:
        raise ValueError(f"The dimension '{dimension}' has no rows.")

    # Initialize the denromalized table with the highest level as the key
    if max_level == 1:
        denormalized_dimension = dimension_hierarchy[["code", "description"]].rename(
            columns={"code": dimension, "description": f"{dimension}_description"}
        )
    else:
        denormalized_dimension = dimension_hierarchy[
            dimension_hierarchy["level"] == max_level
        ][["code", "description"]].rename(
            columns={
                "code": f"{dimension}_{max_level}",
                "description": f"{dimension}_{max_level}_description",
            }
        )

    # Loop from max level down to 2 to create the mapping correctly
    for level in range(max_level, 1, -1):
        # Create the parent map where current level's "code" maps to the previous level's "parent"
        parent_map = dimension_hierarchy[dimension_hierarchy["level"] == level][
            ["code", "parent"]
        ].rename(
            columns={"code": f"{dimension}_{level}", "parent": f"{dimension}_{level-1}"}
        )
        # add parent column's description
        parent_map = pd.merge(
            parent_map,
            dimension_hierarchy[["code", "description"]],
            left_on=f"{dimension}_{level-1}",
            right_on="code",
        )
        parent_map.drop(columns=["code"], inplace=True)
        parent_map.rename(
            columns={"description": f"{dimension}_{level-1}_description"}, inplace=True
        )
        denormalized_dimension = pd.merge(
            denormalized_dimension,
            parent_map,
            left_on=f"{dimension}_{level}",
            right_on=f"{dimension}_{level}",
        )

    return denormalized_dimension


def balance_hierarchy_with_reserve_codes(hierarchy_df):
    new_rows = []
    node_mapping = {}
    hierarchy_df = hierarchy_df.sort_values(
        by=["dimension", "level", "code"]
    ).reset_index(drop=True)

    for dimension in hierarchy_df["dimension"].unique():
        dimension_df = hierarchy_df[hierarchy_df["dimension"] == dimension].copy()
        max_level = dimension_df["level"].max()
        existing_codes = set(dimension_df["code"].values)
        all_codes = set(dimension_df["code"])

        # Initialize mapping for the current dimension
        node_mapping[dimension] = {}

        for code in all_codes:
            code_level = dimension_df.loc[
                dimension_df["code"] == code, "level"
            ].iloc[0]

            # Skip nodes with level == 0
            if code_level == 0:
                continue

            current_code = code
            original_code = code

            # If code_level < max_level, add reserve codes
            if code_level < max_level:
                for level in range(code_level + 1, max_level + 1):
                    reserve_code = f"{original_code}_Reserve_{level}"
                    new_row = {
                        "dimension": dimension,
                        "level": level,
                        "code": reserve_code,
                        "parent": current_code,
                        "description": "<Reserve>",
                    }
                    new_rows.append(new_row)
                    existing_codes.add(reserve_code)
                    current_code = reserve_code
                
                node_mapping[dimension][original_code] = current_code

    # Convert new_rows to DataFrame
    if new_rows:
        reserve_df = pd.DataFrame(new_rows)
    else:
        reserve_df = pd.DataFrame(columns=hierarchy_df.columns)

    # Concatenate the original dataframe with the new reserve codes
    result_df = pd.concat([hierarchy_df, reserve_df], ignore_index=True)

    # Sort by dimension, level, and code for the final output
    result_df = result_df.sort_values(by=["dimension", "level", "code"]).reset_index(
        drop=True
    )

    return result_df, node_mapping


def create_flat_fact_table(atomic_fact_table, hierarchy):
    """
    Flattens the atomic_fact_table by joining with dimension tables based on the hierarchy.
    Uses inner joins while performing pre-validation to ensure all keys are present and unique.
    Remaps intermediate leaf nodes to new balanced leaf nodes as needed.
    If issues are detected, missing keys or duplicate join keys are written to disk and an error is raised.

    Parameters:
    - atomic_fact_table (pd.DataFrame): The fact table to be flattened.
    - hierarchy (pd.DataFrame): The hierarchy defining dimension levels.

    Returns:
    - pd.DataFrame: The flattened fact table.

    Raises:
    - ValueError: If missing joins or duplicate join keys are detected.
    - KeyError: If the join key is not found in the flattened dimension.
    """
    print("\n-----------Create flat fact table-----------")
    atomic_fact_table = atomic_fact_table.copy()

    # Add reserve codes to the hierarchy
    balanced_hierarchy, reserve_mapping = balance_hierarchy_with_reserve_codes(hierarchy)

    # Remap intermediate nodes to balanced reserve nodes
    if reserve_mapping:
        for dim in reserve_mapping:
            if dim in atomic_fact_table.columns:
                mapping = reserve_mapping[dim]
                atomic_fact_table[dim] = (
                    atomic_fact_table[dim].map(mapping).fillna(atomic_fact_table[dim])
                )

    dimensions = balanced_hierarchy["dimension"].unique()

    # Keep track of the initial row count for final verification
    initial_row_count = len(atomic_fact_table)
    print(f"Initial row count: {initial_row_count}")

    # Iterate over each dimension to perform joins
    for dim in dimensions:
        print(f"Processing dimension: '{dim}'")

        # Create the denormalized dimension table and get the leaf mapping
        flattened_dimension = denormalize_balanced_dimension(balanced_hierarchy, dim)

        # Determine the max level for the current dimension
        max_level = balanced_hierarchy[balanced_hierarchy["dimension"] == dim][
            "level"
        ].max()

        # Define join_key based on the number of levels
        join_key = f"{dim}_{max_level}" if max_level > 1 else dim

        # Ensure join_key exists in the flattened_dimension
        if join_key not in flattened_dimension.columns:
            raise KeyError(
                f"Join key '{join_key}' not found in flattened dimension '{dim}'."
            )

        # Check and align data types between fact table and flattened dimension
        fact_dtype = atomic_fact_table[dim].dtype
        dim_dtype = flattened_dimension[join_key].dtype
        if fact_dtype != dim_dtype:
            print(
                f" - Casting '{dim}' from {fact_dtype} to {dim_dtype} to match join key."
            )
            atomic_fact_table[dim] = atomic_fact_table[dim].astype(dim_dtype)

        # Check for duplicate keys in the flattened_dimension
        duplicates = flattened_dimension[join_key].duplicated().sum()
        if duplicates > 0:
            print(
                f" - WARNING: {duplicates} duplicate keys found in flattened dimension '{dim}'."
            )
            # Write duplicated join keys to CSV for inspection
            duplicated_keys = flattened_dimension[
                flattened_dimension[join_key].duplicated(keep=False)
            ][join_key].unique()
            duplicated_df = pd.DataFrame({join_key: duplicated_keys})
            duplicated_df.to_csv(f"data/duplicate_join_keys_{dim}.csv", index=False)
            raise ValueError(
                f"Duplicate join keys found in dimension '{dim}'. See 'data/duplicate_join_keys_{dim}.csv' for details."
            )

        # Pre-validation: Ensure all keys in fact table exist in flattened_dimension
        unique_fact_keys = atomic_fact_table[[dim]].drop_duplicates()
        unique_dim_keys = flattened_dimension[[join_key]].drop_duplicates()

        # Perform a left join on unique keys to identify missing joins
        validation_merge = pd.merge(
            unique_fact_keys,
            unique_dim_keys,
            left_on=dim,
            right_on=join_key,
            how="left",
        )

        # Identify missing joins where join_key is NaN
        missing_joins = validation_merge[join_key].isnull()
        missing_count = missing_joins.sum()

        if missing_count > 0:
            print(f" - ERROR: {missing_count} keys are missing in dimension '{dim}'.")
            # Extract the missing keys
            missing_keys = validation_merge.loc[missing_joins, dim].unique()
            # Save missing keys to CSV for inspection
            missing_keys_df = pd.DataFrame({dim: missing_keys})
            missing_keys_df.to_csv(f"data/missing_keys_{dim}.csv", index=False)
            raise ValueError(
                f"Missing join keys detected in dimension '{dim}'. See 'data/missing_keys_{dim}.csv' for details."
            )
        else:
            print(
                f" - All join keys are present in dimension '{dim}'. Proceeding with remapping and inner join."
            )

        # Perform the inner join as in the original function
        before_merge = len(atomic_fact_table)
        atomic_fact_table = pd.merge(
            atomic_fact_table,
            flattened_dimension,
            left_on=dim,
            right_on=join_key,
            how="inner",
        )
        after_merge = len(atomic_fact_table)
        print(
            f" - Row count before inner join: {before_merge}, after inner join: {after_merge}"
        )

        # Verify that row count remains the same
        if after_merge != before_merge:
            discrepancy = after_merge - before_merge
            print(
                f" - WARNING: Row count changed by {discrepancy} after inner join on dimension '{dim}'."
            )
            # Save the join key causing discrepancy
            # To identify the problematic keys, perform an anti-join
            merged_keys = set(atomic_fact_table[join_key])
            original_keys = set(unique_fact_keys[dim])
            missing_after_join = original_keys - merged_keys
            if missing_after_join:
                missing_after_df = pd.DataFrame({dim: list(missing_after_join)})
                missing_after_df.to_csv(
                    f"data/post_join_missing_keys_{dim}.csv", index=False
                )
                print(
                    f" - Missing keys after join saved to 'data/post_join_missing_keys_{dim}.csv'."
                )
            raise ValueError(
                f"Row count changed after inner join on dimension '{dim}'. See 'data/post_join_missing_keys_{dim}.csv' for details."
            )
        else:
            print(f" - Row count verification passed for dimension '{dim}'.")

        # Drop the original dimension column as it's now been joined
        atomic_fact_table = atomic_fact_table.drop(columns=[dim])

    # Rearrange columns to have 'value' at the end
    cols = [col for col in atomic_fact_table.columns if col != "value"] + ["value"]
    atomic_fact_table = atomic_fact_table[cols]

    final_row_count = len(atomic_fact_table)
    print(f"\nFinal row count: {final_row_count}")

    # Final verification: Ensure that the final row count matches the initial row count
    if final_row_count != initial_row_count:
        discrepancy = final_row_count - initial_row_count
        print(
            f" - ERROR: Final row count ({final_row_count}) does not match initial row count ({initial_row_count})."
        )
        raise ValueError(
            f"Final row count mismatch: {discrepancy} rows {'added' if discrepancy > 0 else 'removed'}."
        )
    else:
        print(
            " - Final row count verification passed. No records were lost or duplicated."
        )

    return atomic_fact_table


def create_atomic_facts_table(facts_with_summaries, hierarchy_df, reserve_tolerance):
    # The key insight is that for everything we add, we also subtract it back out.
    # The only exception is the top level row which represents everything. But once we have that
    # everything else is just filling in details and wont change our totals.
    # To achive this I follow the insight behind inclusion-exlucion principle in set theory.
    # For 3 sets that is: ∣A∪B∪C∣=∣A∣+∣B∣+∣C∣−∣A∩B∣−∣A∩C∣−∣B∩C∣+∣A∩B∩C∣
    # The combinatorial application of that uses the parent/child relationships.
    # For 3 dimensions, each with a parent, this would add 8 records, like so:
    # +[A-Child, B-Child, C-Child] (the original record--next we neutralize it with 7 more records),
    # -[A-Parent, B-Child, C-Child], -[A-Child, B-Parent, C-Child], -[A-Child, B-Parent, C-Child],
    # +[A-Parent, B-Parent, C-Child], +[A-Child, B-Parent, C-Parent], +[A-Parent, B-Child, C-Parent],
    # -[A-Parent, B-Parent, C-Parent]. You can see how they are sort of the same.
    # The beauty is that the value in all 8 cases is the same, making it a very simple calculuation.

    # The obvious downside is the potential for data explosion. This will multiply the data set by
    # 2^n where n is the number of dimensions with parent-child structure.
    # To avoid the possibility of running out of memory in trying to do this, I split it into batches
    # based on the parameter internal_max_row_count_size parameter

    # Note another key thing is that we are at the same time dropping to the "atomic" level.
    # We will use the naming convention {member_name}_reserve_{level} to move an item to the lowest level.

    internal_max_row_count_size = 1_000_000

    result_df = facts_with_summaries.copy()
    dimensions = hierarchy_df["dimension"].unique().tolist()
    hierarchy_mappings = build_hierarchy_mappings(hierarchy_df, True)

    # Identify the "double dimensions" (those with non-empty hierarchy mappings. Each double dimension will double the size of the result set.)
    doubled_dimensions = [
        dimension for dimension in dimensions if not hierarchy_mappings[dimension].empty
    ]
    n_double_dimensions = len(doubled_dimensions)

    # Estimate the number of rows after the explosion
    estimated_intermediate_size = len(result_df) * (2**n_double_dimensions)

    # Calculate the number of batches based on the internal batch size
    num_batches = max(
        1, int(np.ceil(estimated_intermediate_size / internal_max_row_count_size))
    )

    # Calculate the number of rows per batch based on num_batches
    batch_size = max(1, int(len(result_df) / num_batches))

    final_results = []
    intermediate_record_count = 0
    initial_record_count = len(result_df)

    # Process the DataFrame in chunks by index slicing
    for i in range(0, len(result_df), batch_size):
        batch = result_df.iloc[i : i + batch_size]

        # Process each dimension in the batch
        for dimension in doubled_dimensions:
            dim_df = hierarchy_mappings[dimension]

            # Merge the batch with the hierarchy mappings for the current dimension
            batch = batch.merge(
                dim_df, left_on=dimension, right_on="prior_member", how="inner"
            )

            # Rename columns to avoid conflicts
            batch.rename(columns={dimension: f"{dimension}_prior"}, inplace=True)
            batch.rename(
                columns={
                    "new_member": dimension,
                    "new_member_type": f"{dimension}_type",
                },
                inplace=True,
            )
            # Drop columns related to prior member and level
            batch.drop(columns=["prior_member", "level"], inplace=True)

        # Count the parent records in each row
        batch["parent_count"] = batch[
            [f"{dimension}_type" for dimension in doubled_dimensions]
        ].apply(lambda row: (row == "parent").sum(), axis=1)

        # Apply the balancing mechanism (multiply value by -1 if odd parent count)
        batch.loc[batch["parent_count"] % 2 == 1, "value"] *= -1

        # Track intermediate record count
        intermediate_record_count += len(batch)

        # Group by dimensions and sum values for this batch
        batch = batch.groupby(dimensions)["value"].sum().reset_index()

        # Append batch results
        final_results.append(batch)

    # Concatenate all batch results
    concatenated_df = pd.concat(final_results)

    # Final aggregation across all batches
    final_df: pd.DataFrame
    final_df = concatenated_df.groupby(dimensions)["value"].sum().reset_index()

    # Apply the reserve tolerance filter. 
    # may want to think a bit about this one more. or maybe it is more about how it is used
    # I am not sure the tolerance for identifyin errors should be the same as the tolerance
    # for removing very small values.
    final_df = final_df[abs(final_df["value"]) > reserve_tolerance]
    final_record_count = len(final_df)

    # Output the counts and ratios, including the inflation ratio
    print(f"\n----------Create atomic facts-------------")
    print(f"Initial record count: {initial_record_count}")
    print(f"Intermediate record count: {intermediate_record_count}")
    print(f"Final record count: {final_record_count}")

    # Calculate and print the inflation and reduction ratios
    if initial_record_count > 0:
        inflation_ratio = intermediate_record_count / initial_record_count
        print(f"Inflation Ratio (intermediate/initial): {inflation_ratio:.2f}")

    if final_record_count > 0:
        reduction_ratio = intermediate_record_count / final_record_count
        print(f"Reduction Ratio (intermediate/final): {reduction_ratio:.2f}")

    return final_df, hierarchy_df


def recalc_hierarchy_levels(hierarchy_df):
    hierarchy_df["level"] = hierarchy_df.apply(
        lambda row: 0 if row["parent"] is None else None, axis=1 #xxx downstream issue identified with na records when importing from pandas. determine how best to solve. work around check pd.isna.
    )

    def set_level(parent_code, level):
        hierarchy_df.loc[hierarchy_df["parent"] == parent_code, "level"] = int(
            level
        )  # level
        children = hierarchy_df[hierarchy_df["parent"] == parent_code]["code"]
        for child in children:
            set_level(child, level + 1)

    top_level_nodes = hierarchy_df[hierarchy_df["parent"].isnull()]["code"]
    for node in top_level_nodes:
        set_level(node, 1)

    #raise error if nulls in level
    if hierarchy_df["level"].isnull().any():
        raise ValueError("Null values found in the 'level' column.")

    hierarchy_df["level"] = hierarchy_df["level"].astype(int)
    return hierarchy_df


def validate_and_define_dimensions(df):
    # Validate and map dimensions
    dim_mapping = []
    error_messages = []

    if "value" not in df.columns:
        error_messages.append(
            "Required column 'value' is missing in the incoming data."
        )

    for col in df.columns:
        if "filterDim" in col and "_code" in col:
            base_name = col.replace("_code", "").replace("_description", "")
            code_column = f"{base_name}_code"
            description_column = f"{base_name}_description"
            if code_column not in df.columns or description_column not in df.columns:
                error_messages.append(
                    f"Filter dimension '{base_name}' is missing either '{code_column}' or '{description_column}'."
                )
            else:
                dim_mapping.append(
                    {
                        "base_name": base_name,
                        "code_column": code_column,
                        "description_column": description_column,
                        "type": "filter",
                    }
                )

        if "rowsDim" in col and "_code" in col:
            base_name = (
                col.replace("_code", "")
                .replace("_description", "")
                .replace("_rel_level", "")
            )
            code_column = f"{base_name}_code"
            description_column = f"{base_name}_description"
            rel_level_column = f"{base_name}_rel_level"

            if (
                code_column not in df.columns
                or description_column not in df.columns
                or rel_level_column not in df.columns
            ):
                error_messages.append(
                    f"Row dimension '{base_name}' is missing either '{code_column}', '{description_column}', or '{rel_level_column}'."
                )
            else:
                dim_mapping.append(
                    {
                        "base_name": base_name,
                        "code_column": code_column,
                        "description_column": description_column,
                        "rel_level_column": rel_level_column,
                        "type": "row",
                    }
                )

    if error_messages:
        raise ValueError("\n".join(error_messages))

    return pd.DataFrame(dim_mapping)


def child_parent_records(unique_df, mapping_row):
    # helper function for add_to_hierarchy
    """
    For a given DataFrame of unique dimension members and a mapping_row, generates a DataFrame
    of child-parent records.

    The input DataFrame is assumed to have a column for the dimension code, description, and
    rel_level. The mapping_row is a tuple with the column names for these columns.

    The function first adds a new rel_level column to represent the parent level of each node,
    then performs a self-merge to align each node with potential parent nodes. The result is
    filtered to only include parents with a lower row_number (i.e. above the current node in
    the original DataFrame), and then sorted by node and parent. The highest (i.e. most recent
    if going down the table) parent is selected for each node, and the resulting DataFrame is
    sorted by node and parent, and then grouped by node. That is the node's parent.

    The final DataFrame has columns for the child code, description, and parent code, and is
    sorted by child code.

    Parameters
    ----------
    unique_df : DataFrame
        DataFrame of unique dimension members
    mapping_row : tuple
        Tuple with column names for the dimension code, description, and rel_level

    Returns
    -------
    DataFrame
        DataFrame of child-parent records

    """

    # Adjust the levels in the DataFrame for the merge

    unique_df[f"{mapping_row.rel_level_column}_parent"] = (
        unique_df[mapping_row.rel_level_column] - 1
    )

    #xxx reported slow performance. evidently moving to a row based approach and finding the next parent is faster
    # Get unique levels in the dataframe
    unique_levels = unique_df[mapping_row.rel_level_column].unique()

    for level in unique_levels:
        if (level - 1) not in unique_levels and level > 1:
            raise ValueError(f"No rows with level {level - 1} found for level {level} for {mapping_row.rel_level_column} in the data.")
    

    # Perform the self-merge to align each node with potential parent nodes
    potential_parents = unique_df.merge(
        unique_df[
            [
                mapping_row.code_column,
                mapping_row.rel_level_column,
                "row_number",
                mapping_row.description_column,
            ]
        ],
        left_on=f"{mapping_row.rel_level_column}_parent",
        right_on=mapping_row.rel_level_column,
        suffixes=("", "_parent"),
    )

    potential_parents = potential_parents[
        potential_parents["row_number"] < potential_parents["row_number_parent"]
    ]

    # Sort by 'row_id' and 'row_id_parent' ascending
    potential_parents = potential_parents.sort_values(
        by=["row_number", "row_number_parent"]
    )
    # Group by the node and select the first parent within the sorted group
    child_parents = (
        potential_parents.groupby(mapping_row.code_column).first().reset_index()
    )

    # Select & rename the relevant columns
    child_parents = child_parents[
        [
            mapping_row.code_column,
            mapping_row.description_column,
            f"{mapping_row.code_column}_parent",
        ]
    ]
    child_parents.columns = ["code", "description", "parent"]
    return child_parents


def add_to_hierarchy(incoming_df, mapping_df, hierarchy_df):
    incoming_df = incoming_df.copy()
    incoming_df["row_number"] = incoming_df.index + 1
    #xxx reported slow performance. Hoping switching to duckdb resolves
    for mapping_row in mapping_df[mapping_df["type"] == "row"].itertuples():
        result = child_parent_records(incoming_df, mapping_row)
        if hierarchy_df.empty:
            hierarchy_df = result
            hierarchy_df["dimension"] = mapping_row.dimension_name
        else:
            new_nodes = (
                result.merge(
                    hierarchy_df["code"], on="code", how="left", indicator=True
                )
                .loc[lambda x: x["_merge"] == "left_only"]
                .drop(columns=["_merge"])
            )
            new_nodes["dimension"] = mapping_row.dimension_name
            hierarchy_df = pd.concat([hierarchy_df, new_nodes], ignore_index=True)

    hierarchy_df = recalc_hierarchy_levels(hierarchy_df)
    return hierarchy_df


def update_columns_with_dimension_names(incoming_data, dim_mapping_df):
    incoming_data = incoming_data.copy()
    for mapping_row in dim_mapping_df.itertuples():
        incoming_data = incoming_data.rename(
            columns={mapping_row.code_column: mapping_row.dimension_name}
        )
        incoming_data = incoming_data.drop(columns=[mapping_row.description_column])
        if mapping_row.type == "row":
            incoming_data = incoming_data.drop(columns=[mapping_row.rel_level_column])
    return incoming_data


def parse_data(incoming_data: pd.DataFrame, hierarchy_df: pd.DataFrame):
    dim_mapping_df = validate_and_define_dimensions(incoming_data)
    is_valid, dim_mapping_df = incremental_assign_dimension_names(
        incoming_data, hierarchy_df, dim_mapping_df
    )
    if not is_valid:
        raise ValueError("Data is not valid")

    hierarchy_df = add_to_hierarchy(incoming_data, dim_mapping_df, hierarchy_df)
    incremental_fact_data = update_columns_with_dimension_names(incoming_data, dim_mapping_df)
    return incremental_fact_data, hierarchy_df


def merge_atomic_facts(
    atomic_facts_df: pd.DataFrame, 
    incremental_atomic_data: pd.DataFrame, 
    reserve_tolerance: float
) -> pd.DataFrame:

    # Identify key columns (all columns except 'value')
    key_columns = atomic_facts_df.columns.difference(["value"])

    # Concatenate the old atomic cube with the incremental data
    combined_df = pd.concat(
        [atomic_facts_df, incremental_atomic_data], ignore_index=True
    )

    # Group by key columns and sum the 'value' column
    grouped_df: pd.DataFrame = (
        combined_df.groupby(list(key_columns)).agg({"value": "sum"}).reset_index()
    )

    # Remove records with absolute value below tolerance. 
    # see comment above in function create_atomic_facts_table
    grouped_df = grouped_df[abs(grouped_df["value"]) > reserve_tolerance]
    return grouped_df


def establish_founding_data_sets(incoming_data, reserve_tolerance):
    ##Figure out the dimensions
    dim_mapping_df = validate_and_define_dimensions(incoming_data)
    dim_mapping_df = add_dimension_names(dim_mapping_df, incoming_data, True)

    ##Create initial hierarchy
    # create category roots
    # -- creates abstract, top-level nodes for each dimension to serve as organizational anchors for the dimension
    top_level_nodes = [
        {
            "dimension": name,
            "code": f"Top_{name}",
            "description": f"Top node for the {name} dimension",
            "parent": None,
        }
        for name in dim_mapping_df["dimension_name"]
    ]
    hierarchy_df = pd.DataFrame(top_level_nodes)
    hierarchy_df["dimension"] = hierarchy_df["dimension"].astype("category")
    # add hierarchies
    hierarchy_df = add_first_hierarchy_level(incoming_data, dim_mapping_df, hierarchy_df)
    hierarchy_df = add_to_hierarchy(incoming_data, dim_mapping_df, hierarchy_df)
    
    ##Create all facts table
    all_facts_df = update_columns_with_dimension_names(incoming_data, dim_mapping_df)
    
    ## Validate
    # Ensure all_facts_df has all required facts
    missing_rows_all_ancestors(all_facts_df, hierarchy_df)

    ##Create atomic facts table
    atomic_facts_df, hierarchy_df = create_atomic_facts_table(
        all_facts_df, hierarchy_df, reserve_tolerance
    )

    ##Flatten hierarchy
    flattened_df = create_flat_fact_table(atomic_facts_df, hierarchy_df)

    print("Founding data set processing completed successfully.")
    return all_facts_df, hierarchy_df, atomic_facts_df, flattened_df


def incremental_add(
    all_facts_df: pd.DataFrame,
    hierarchy_df: pd.DataFrame,
    atomic_facts_df: pd.DataFrame,
    incremental_data: pd.DataFrame,
    reserve_tolerance: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    # Parse the data into Incremental_Fact_Summary.
    # hierarchy_df returned is the full hierarchy with new members added
    incremental_facts_df: pd.DataFrame
    incremental_facts_df, hierarchy_df = parse_data(
        incremental_data, hierarchy_df
    )

    # check for missing rows
    # process will fail if invalid
    missing_rows_all_ancestors(incremental_facts_df, hierarchy_df)

    # validate incremental fact summary
    # merge new facts into all_facts_df. 
    # Let incremental_facts_df be just the truly new facts.
    # process will fail if invalid
    all_facts_df, incremental_facts_df = validate_and_merge_new_facts(
        all_facts_df, incremental_facts_df, reserve_tolerance
    )

    # Generate incremental atomic data
    incremental_atomic_data: pd.DataFrame
    hierarchy_with_reserves: pd.DataFrame
    incremental_atomic_data, hierarchy_with_reserves = create_atomic_facts_table(incremental_facts_df, hierarchy_df, reserve_tolerance)

    # Merge into Atomic_Fact_Cube
    atomic_facts_df = merge_atomic_facts(atomic_facts_df, incremental_atomic_data, reserve_tolerance)

    # create_flat_fact_table
    flattened_df = create_flat_fact_table(atomic_facts_df, hierarchy_with_reserves)

    return all_facts_df, hierarchy_df, atomic_facts_df, flattened_df


def sort_child_to_parent(facts, hierarchy):
    sorted_orders = {}

    # Build a hierarchy tree for each dimension using adjacency lists
    dimensions = hierarchy["dimension"].unique()

    for dimension in dimensions:
        # Filter hierarchy for the current dimension
        dim_hierarchy = hierarchy[hierarchy["dimension"] == dimension]

        # Initialize the tree as a dictionary for parent -> children relationships
        tree = {}

        # Populate the tree with parent-child relationships
        for _, row in dim_hierarchy.iterrows():
            parent = row["parent"]
            code = row["code"]
            # If the parent is None (i.e., top-level node), add it to the tree with an empty list of children
            if parent not in tree:
                tree[parent] = []
            # Append the current code (child) to the parent's list of children
            tree[parent].append(code)

            # Ensure the current code also exists in the tree, even if it has no children
            if code not in tree:
                tree[code] = []

        # Perform DFS to get the order for each dimension
        sorted_codes = []

        def dfs(node):
            #ensure parents come AFTER their children
            # Recursively apply DFS on all children of the current node
            for child in tree[node]:
                dfs(child)
            # Append the node to the sorted list
            sorted_codes.append(node)

        # Identify top-level parents (nodes where parent is NaN)
        top_parents = dim_hierarchy[pd.isna(dim_hierarchy["parent"])]["code"].tolist()

        # Start DFS traversal from all top-level parents
        for parent in top_parents:
            dfs(parent)

        # Create a mapping of code to rank (based on the DFS order)
        sorted_orders[dimension] = {code: i for i, code in enumerate(sorted_codes)}

    # Replace the codes in the facts table with their ranks
    for dimension in dimensions:
        # Map ranks using the order from sorted_orders
        facts[f"{dimension}_rank"] = facts[dimension].map(sorted_orders[dimension])

        # Fill any NaN ranks (if they exist) with the maximum rank + 1 to ensure proper sorting
        facts[f"{dimension}_rank"].fillna(
            facts[f"{dimension}_rank"].max() + 1, inplace=True
        )

    # Sort the facts table by the rank columns and drop the rank columns after sorting
    rank_columns = [f"{dimension}_rank" for dimension in dimensions]
    sorted_facts = facts.sort_values(by=rank_columns).drop(columns=rank_columns)
    return sorted_facts


def prepare_query_set(hierarchy_df, requested_members):
    """
    Prepares a query dataset based on requested members and the given hierarchy,
    ensuring it respects parent-child relationships. Uses DuckDB for processing.

    Parameters:
    - hierarchy_df: DataFrame representing the hierarchy of dimensions.
    - requested_members: Dictionary where each key is a dimension name, and the value is a list of member codes.

    Returns:
    - A DataFrame suitable for querying the complete dataset.
    """

    # Generate a unique table name for this function call
    table_name = f"hierarchy_{uuid.uuid4().hex[:6]}"

    try:
        # Load hierarchy into DuckDB
        conn.execute(f"DROP TABLE IF EXISTS {table_name};")
        conn.execute(f"CREATE TABLE {table_name} AS SELECT * FROM hierarchy_df;")

        # Helper functions
        def get_top_level_members(dim):
            query = f"""
            SELECT code
            FROM {table_name}
            WHERE dimension = '{dim}' AND level = 1;
            """
            return [row[0] for row in conn.execute(query).fetchall()]

        def get_ancestors(dim, members):
            query = f"""
            WITH RECURSIVE ancestors AS (
                SELECT code, parent, description, level, CAST(code AS VARCHAR) AS path
                FROM {table_name}
                WHERE dimension = '{dim}' AND code IN ({','.join([f"'{m}'" for m in members])})
                UNION ALL
                SELECT h.code, h.parent, h.description, h.level, CONCAT(a.path, '->', h.code) AS path
                FROM {table_name} h
                INNER JOIN ancestors a ON h.code = a.parent
            )
            SELECT DISTINCT code, description, level, path
            FROM ancestors
            WHERE level != 0
            ORDER BY path DESC; -- Depth-first ordering ensures children come before parents
            """
            return pd.DataFrame(conn.execute(query).fetchall(), columns=["code", "description", "rel_level", "path"])

        # Separate filter and row dimensions
        filter_dims, row_dims = {}, {}

        for dim, members in requested_members.items():
            if len(members) == 1:
                filter_dims[dim] = members[0]
            else:
                row_dims[dim] = members

        # Handle unlisted dimensions as filter dimensions with top-level members
        listed_dims = set(requested_members.keys())
        for dim in hierarchy_df["dimension"].unique():
            if dim not in listed_dims:
                top_level_members = get_top_level_members(dim)
                if len(top_level_members) == 1:
                    filter_dims[dim] = top_level_members[0]
                else:
                    row_dims[dim] = top_level_members

        # Prepare filter dimensions
        filter_data = {}
        filter_count = 1

        for dim, member in filter_dims.items():
            query = f"""
            SELECT code AS filterDim{filter_count}_code,
                   description AS filterDim{filter_count}_description
            FROM {table_name}
            WHERE dimension = '{dim}' AND code = '{member}';
            """
            filter_data[dim] = pd.DataFrame(conn.execute(query).fetchall(),
                                            columns=[f"filterDim{filter_count}_code", f"filterDim{filter_count}_description"])
            filter_count += 1

        # Prepare row dimensions
        row_data = {}
        row_count = 1

        for dim, members in row_dims.items():
            ancestors_df = get_ancestors(dim, members)
            ancestors_df = ancestors_df.sort_values("path", ascending=False)  # Sort using the depth-first path
            ancestors_df.drop(columns=["path"], inplace=True)  # Remove the path column after sorting
            ancestors_df.columns = [f"rowsDim{row_count}_code", f"rowsDim{row_count}_description", f"rowsDim{row_count}_rel_level"]
            row_data[dim] = ancestors_df
            row_count += 1

        # Cartesian product of all dimensions
        all_dims = list(filter_data.values()) + list(row_data.values())

        # Generate tuples for each DataFrame
        tuple_dataframes = []
        for df in all_dims:
            tuple_dataframes.append(df.itertuples(index=False, name=None))

        # Perform cartesian product
        cartesian_product = list(product(*tuple_dataframes))

        # Flatten column names for the final DataFrame
        column_names = []
        for df in all_dims:
            column_names.extend(df.columns)

        flattened_cartesian_product = [tuple(item for sublist in t for item in sublist) for t in cartesian_product]

        # Create the final DataFrame
        result = pd.DataFrame(flattened_cartesian_product, columns=column_names)

        return result
    finally:
        # Clean up: Drop the temporary table
        conn.execute(f"DROP TABLE IF EXISTS {table_name};")
