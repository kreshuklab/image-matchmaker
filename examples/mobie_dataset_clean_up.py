import json

# clean up dataset.json
input_path = "./data/test/mobie_project/platy1_muscles_stardist/dataset.json"

# Define the key you want to delete
key_to_delete = "platy1_muscles_stardist_moving_prealigned_rigid_aligned"

# Load the JSON file
with open(input_path, "r") as file:
    dataset_dict = json.load(file)


# Recursively delete all entries with the specified key
def delete_key_recursively(obj, key):
    if isinstance(obj, dict):
        # Remove the key if it exists in the current dictionary
        obj.pop(key, None)
        # Recurse into the dictionary values
        for value in obj.values():
            delete_key_recursively(value, key)
    elif isinstance(obj, list):
        # Recurse into each item in the list
        for item in obj:
            delete_key_recursively(item, key)


# Call the recursive function on the loaded JSON data
delete_key_recursively(dataset_dict, key_to_delete)

# Save the modified JSON back to a file
with open(input_path, "w") as file:
    json.dump(dataset_dict, file, indent=4)

print(f"Entries with key '{key_to_delete}' have been deleted.")
