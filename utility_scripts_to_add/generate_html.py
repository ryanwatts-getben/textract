import json
import os
from jinja2 import Environment, FileSystemLoader
from datetime import datetime

def sanitize_filename(filename):
    # Replace any characters that are invalid in filenames
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, '_')
    return filename

def parse_date(date_string):
    # Try different date formats
    date_formats = ['%Y-%m-%d', '%m/%d/%y', '%m/%d/%Y', '%Y/%m/%d']
    for fmt in date_formats:
        try:
            return datetime.strptime(date_string, fmt).strftime('%Y-%m-%d')
        except ValueError:
            continue
    return 'Unknown_Date'

def process_lab_results(data):
    if isinstance(data, dict):
        for key, value in data.items():
            if key in ['Chemistry', 'LabResults']:
                processed_results = {}
                for test, result in value.items():
                    if isinstance(result, str):
                        parts = result.split()
                        if len(parts) > 1:
                            processed_results[test] = {
                                'Result': parts[0],
                                'Range': ' '.join(parts[1:])
                            }
                        else:
                            processed_results[test] = {'Result': result, 'Range': ''}
                    elif isinstance(result, dict):
                        processed_results[test] = result
                data[key] = processed_results
            elif isinstance(value, (dict, list)):
                process_lab_results(value)
    elif isinstance(data, list):
        for item in data:
            process_lab_results(item)

def main():
    # Define paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    json_dir = os.path.join(current_dir, '..', 'htmlcomplete')
    template_dir = os.path.join(current_dir, '..', 'htmlcomplete', 'templates')
    output_dir = os.path.join(json_dir, 'output')

    # Set up Jinja2 environment
    env = Environment(loader=FileSystemLoader(template_dir))
    template = env.get_template('record_template.html')

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Iterate over each JSON file in the directory
    for filename in os.listdir(json_dir):
        if filename.endswith('.json'):
            json_path = os.path.join(json_dir, filename)
            try:
                with open(json_path, 'r', encoding='utf-8') as file:
                    data = json.load(file)

                # Extract necessary data from JSON
                date = parse_date(data.get('Date', 'Unknown Date'))
                patient_info = data.get('PatientInformation', {})
                codes = data.get('Codes', {})
                procedures_or_findings = data.get('ProceduresOrFindings', [])
                daily_financial_summary = data.get('DailyFinancialSummary', [])
                other_information = data.get('OtherInformation', [])

                # Process lab results
                process_lab_results(procedures_or_findings)

                # Render the HTML using the template
                rendered_html = template.render(
                    date=date,
                    patient_information=patient_info,
                    codes=codes,
                    procedures_or_findings=procedures_or_findings,
                    daily_financial_summary=daily_financial_summary,
                    other_information=other_information
                )

                # Define output HTML file path
                safe_filename = sanitize_filename(f"{date}.html")
                output_file = os.path.join(output_dir, safe_filename)
                with open(output_file, 'w', encoding='utf-8') as out_file:
                    out_file.write(rendered_html)

                print(f"Generated HTML for {date}: {output_file}")
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")

if __name__ == "__main__":
    main()