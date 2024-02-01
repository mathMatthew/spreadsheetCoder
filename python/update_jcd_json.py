import os, json
import hashlib

# internal
import jcd


def calculate_hash(file_path):
    """Calculate the MD5 hash of a file."""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def sync_files(directory):
    file_hashes = {}
    problematic_files = []

    # Load existing hash information if available
    hash_file_path = os.path.join(directory, "file_hashes.txt")
    if os.path.exists(hash_file_path):
        with open(hash_file_path, "r") as file:
            for line in file:
                filename, file_hash = line.strip().split(",")
                path = os.path.join(directory, filename)
                if os.path.exists(path):
                    file_hashes[filename] = file_hash

    # Process each JCD file in the directory
    for filename in os.listdir(directory):
        if filename.endswith(".jcd"):
            jcd_path = os.path.join(directory, filename)
            json_filename = filename[:-4] + ".json"
            json_path = os.path.join(directory, json_filename)

            # Calculate current hash of the JCD file
            current_jcd_hash = calculate_hash(jcd_path)

            # Determine if JSON file exists and its hash
            json_exists = os.path.exists(json_path)
            current_json_hash = calculate_hash(json_path) if json_exists else None

            # if both JCD and JSON have changed add to problematic files
            if (
                current_jcd_hash != file_hashes.get(filename)
                and current_json_hash != file_hashes.get(json_filename)
                and json_exists
            ):
                problematic_files.append(filename)
                continue

            old_jcd_hash = file_hashes.get(filename)
            old_json_hash = file_hashes.get(json_filename)

            # if JSON doesn't exist, write new JSON
            if not json_exists:
                jcd.convert_jcd_to_json_file(jcd_path, json_path)
            # if everything is in sync, do nothing.
            elif (
                current_jcd_hash == old_jcd_hash
                and current_json_hash == old_json_hash
                and json_exists
            ):
                continue
            # if JCD has changed, but not JSON, overwrite old JSON with new JSON
            elif (
                current_jcd_hash != old_jcd_hash
                and current_json_hash == old_json_hash
                and json_exists
            ):
                jcd.convert_jcd_to_json_file(jcd_path, json_path)
            # if JSON has changed, but not JCD, overwrite old JCD with new JCD
            elif (
                current_jcd_hash == old_jcd_hash
                and current_json_hash != old_json_hash
                and json_exists
            ):
                jcd.convert_json_to_jcd_file(json_path, jcd_path)

            else:
                raise Exception(f"Could not convert {filename}.")

            # Update the hash information
            file_hashes[filename] = calculate_hash(jcd_path)
            file_hashes[json_filename] = calculate_hash(json_path)

    # Save updated hash information
    with open(hash_file_path, "w") as file:
        for filename, hash_value in file_hashes.items():
            file.write(f"{filename},{hash_value}\n")

    if problematic_files:
        print("WARNING: The following files had issues and were not processed:")
        for file in problematic_files:
            print(file)

def main():
    sync_files("./system_data")

if __name__ == "__main__":
    main()

