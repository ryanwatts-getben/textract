import os

def delete_files_with_strings(directory, strings_to_check):
    try:
        # List all files in the directory
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            
            # Check if the file contains any of the specified strings and is a file
            if any(string in filename.lower() for string in strings_to_check) and os.path.isfile(file_path):
                os.remove(file_path)  # Delete the file
                print(f"Deleted: {file_path}")
            else:
                print(f"Skipped: {file_path}")

    except Exception as e:
        print(f"An error occurred: {e}")

# Define the directory and strings to search for
directory = 'C:/Users/custo/OneDrive/Desktop/medscan/textract/textract/medchrondebug/OrthoSC Records 02.10.24-04.08.24/split_aws/proof'  # Replace with your folder path
strings_to_check = ['clean', 'final', 'details']

# Call the function to delete matching files
delete_files_with_strings(directory, strings_to_check)
