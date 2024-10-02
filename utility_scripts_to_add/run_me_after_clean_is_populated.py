import re
import json
import os
import argparse
from datetime import datetime
from fuzzywuzzy import fuzz

"""
This script should just replace the files in the /clean/ directory, it does not combine the records
"""

def smart_capitalize(name):
    """
    Capitalize each part of the name properly, handling apostrophes and hyphens.
    """
    def capitalize_part(part):
        # Handle names starting with "O'" (e.g., O'Connor)
        if part.lower().startswith("o'"):
            return "O'" + part[2:].capitalize()
        
        # Capitalize each sub-part separated by apostrophes
        sub_parts = part.split("'")
        sub_parts = [sp.capitalize() for sp in sub_parts]
        part = "'".join(sub_parts)
        
        # Capitalize each sub-part separated by hyphens
        sub_parts = part.split("-")
        sub_parts = [sp.capitalize() for sp in sub_parts]
        part = "-".join(sub_parts)
        
        return part

    # Split the name into parts and capitalize each part
    parts = name.split()
    capitalized_parts = [capitalize_part(part) for part in parts]
    return ' '.join(capitalized_parts)

def clean_name(name):
    """
    Clean a given name by removing titles, degrees, salutations, numbers, punctuation,
    single letters, and extra whitespace.
    """
    # List of titles, degrees, and salutations to remove
    titles_degrees_salutations = [
        "MD", "DO", "Dr.", "DR", "Dr", "M.D.", "D.O.", "PhD", "Ph.D.",
        "DDS", "D.D.S.", "DVM", "D.V.M.", "Mr.", "Mrs.", "Miss", "Ms.",
        "Mr", "Mrs", "Miss", "Ms", "APRN", "R1", "MT)", "MD)", "MT)",
        "Md", "md"
    ]
    
    # Remove each title, degree, or salutation from the name
    for title in titles_degrees_salutations:
        name = re.sub(
            r'\b' + re.escape(title) + r'\b|\B' + re.escape(title) + r'\b|^' + re.escape(title) + r'\s|\s' + re.escape(title) + r'$|\s' + re.escape(title) + r'\s',
            ' ', name, flags=re.IGNORECASE
        )
    
    # Reorder name if it contains a comma (assumes "Last, First" format)
    if ',' in name:
        parts = name.split(',')
        if len(parts) >= 2:
            last_name = parts[0].strip()
            first_names = ' '.join(parts[1:]).strip()
            name = f"{first_names} {last_name}"
    
    # Remove punctuation, numbers, single letters, and extra spaces
    name = re.sub(r'[^\w\s\'\-]', '', name)
    name = re.sub(r'\d+', '', name)
    name = re.sub(r'\b[a-zA-Z]\b(?!\')', '', name)
    name = re.sub(r'\s+', ' ', name)
    
    # Capitalize the cleaned name
    name = smart_capitalize(name)
    name = name.strip()
    
    # Reformat name to "First Last" or "First Middle Last"
    parts = name.split()
    if len(parts) == 2:
        first_name, last_name = parts
        name = f"{first_name} {last_name}"
    elif len(parts) > 2:
        first_name = parts[0]
        last_name = parts[-1]
        middle_names = ' '.join(parts[1:-1])
        name = f"{first_name} {middle_names} {last_name}"
    
    return name

def remove_duplicates(names):
    """
    Remove duplicates from a list of names using fuzzy matching with an 80% confidence threshold.
    """
    unique_names = []
    for name in names:
        # Check if the name is similar to any already in the list
        if not any(fuzz.ratio(name, unique_name) > 80 for unique_name in unique_names):
            unique_names.append(name)
    return unique_names

def normalize_date(date_str):
    """
    Normalize a date string to the format YYYY-MM-DD.
    Handles various input formats including:
    - MM/DD/YY, MM-DD-YY
    - MM/DD/YYYY, MM-DD-YYYY
    - YYYY/MM/DD, YYYY-MM-DD
    - DD/MM/YY, DD-MM-YY
    - DD/MM/YYYY, DD-MM-YYYY
    - Month DD, YYYY (e.g., January 01, 2021)
    - DD Month YYYY (e.g., 01 January 2021)
    - Mon DD, YYYY (e.g., Jan 01, 2021)
    - YYYY-MM-DD HH:MM:SS (ISO format with time)
    - MM/DD/YYYY HH:MM:SS
    - DD/MM/YYYY HH:MM:SS
    """
    # List of date formats to try parsing
    date_formats = [
        "%m/%d/%y", "%m-%d-%y",
        "%m/%d/%Y", "%m-%d-%Y",
        "%Y/%m/%d", "%Y-%m-%d",
        "%d/%m/%y", "%d-%m-%y",
        "%d/%m/%Y", "%d-%m-%Y",
        "%B %d, %Y",
        "%d %B %Y",
        "%b %d, %Y",
        "%Y-%m-%d %H:%M:%S",
        "%m/%d/%Y %H:%M:%S",
        "%d/%m/%Y %H:%M:%S"
    ]

    # Clean up the date string
    date_str = date_str.strip().title()

    # Check if the date is already in the desired format
    if re.match(r'^\d{4}-\d{2}-\d{2}$', date_str):
        return date_str

    # Attempt to parse the date using dateutil if available
    try:
        from dateutil import parser
        parsed_date = parser.parse(date_str)
        return parsed_date.strftime("%Y-%m-%d")
    except ImportError:
        pass  # dateutil not available, continue with manual parsing
    except ValueError:
        pass  # dateutil couldn't parse, try manual formats

    # Try each date format to parse the date
    for fmt in date_formats:
        try:
            date_obj = datetime.strptime(date_str, fmt)
            return date_obj.strftime("%Y-%m-%d")
        except ValueError:
            continue

    # Handle special cases for unrecognized formats
    # Handle "YYYYMMDD" format
    if re.match(r'^\d{8}$', date_str):
        try:
            return f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"
        except:
            pass

    # Handle "DD.MM.YYYY" or "MM.DD.YYYY" format
    if re.match(r'^\d{1,2}\.\d{1,2}\.\d{4}$', date_str):
        parts = date_str.split('.')
        if int(parts[1]) <= 12:  # Assume MM.DD.YYYY
            try:
                return datetime(int(parts[2]), int(parts[0]), int(parts[1])).strftime("%Y-%m-%d")
            except ValueError:
                pass
        # If above fails, try DD.MM.YYYY
        try:
            return datetime(int(parts[2]), int(parts[1]), int(parts[0])).strftime("%Y-%m-%d")
        except ValueError:
            pass

    # If all attempts fail, return None
    return None

def process_json_file(input_path, output_path):
    """
    Read a JSON file, clean the names and normalize the date, and write the updated data.
    """
    try:
        # Open and read the JSON data from the input file
        with open(input_path, 'r', encoding='utf-8') as infile:
            data = json.load(infile)
        
        # Process patient information if available
        if "PatientInformation" in data:
            patient_info = data["PatientInformation"]
            
            # Clean the patient's name
            if "PatientFirstNameLastName" in patient_info:
                original_patient_name = patient_info["PatientFirstNameLastName"]
                cleaned_patient_name = clean_name(original_patient_name)
                patient_info["PatientFirstNameLastName"] = cleaned_patient_name
                print(f"Cleaned patient name in {output_path}: {cleaned_patient_name}")
            
            # Clean and deduplicate doctor names
            if "DoctorFirstNameLastName" in patient_info and isinstance(patient_info["DoctorFirstNameLastName"], list):
                original_names = patient_info["DoctorFirstNameLastName"]
                cleaned_names = [clean_name(name) for name in original_names]
                cleaned_names = remove_duplicates(cleaned_names)
                patient_info["DoctorFirstNameLastName"] = cleaned_names
                print(f"Cleaned doctor names in {output_path}: {cleaned_names}")
            
            # Clean and deduplicate referred-to names
            if "ReferredTo" in patient_info and isinstance(patient_info["ReferredTo"], list):
                cleaned_names = [clean_name(name) for name in patient_info["ReferredTo"]]
                patient_info["ReferredTo"] = remove_duplicates(cleaned_names)
            
            # Clean and deduplicate referred-by names
            if "ReferredBy" in patient_info and isinstance(patient_info["ReferredBy"], list):
                cleaned_names = [clean_name(name) for name in patient_info["ReferredBy"]]
                patient_info["ReferredBy"] = remove_duplicates(cleaned_names)
        
        # Normalize the date if available
        if "Date" in data:
            original_date = data["Date"]
            normalized_date = normalize_date(original_date)
            if normalized_date:
                data["Date"] = normalized_date
                print(f"Normalized date in {output_path}: {normalized_date}")
        
        # Write the cleaned and updated data back to the output file
        with open(output_path, 'w', encoding='utf-8') as outfile:
            json.dump(data, outfile, indent=2)
    except Exception as e:
        print(f"Error processing file {input_path}: {e}")

def main():
    """
    Main function to parse command-line arguments and process JSON files.
    """
    # Set up argument parser for command-line input
    parser = argparse.ArgumentParser(description='Clean names and normalize dates in JSON files within a directory.')
    parser.add_argument('directory', help='Directory to scan for JSON files.')
    args = parser.parse_args()
    
    # Define input and output directories
    input_directory = args.directory
    output_directory = os.path.join(input_directory, 'clean_data')
    
    # Create the output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)
    
    # Iterate over all .txt files in the input directory
    for filename in os.listdir(input_directory):
        if filename.endswith('.txt'):
            input_file = os.path.join(input_directory, filename)
            output_file = os.path.join(output_directory, filename)
            
            # Process each file to clean and normalize data
            process_json_file(input_file, output_file)
            print(f"Processed {input_file} -> {output_file}")
    
    print(f"Cleaned files have been saved in: {output_directory}")

if __name__ == '__main__':
    main()