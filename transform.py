import json
import boto3
import logging
import sys
from copy import deepcopy
from typing import Dict, List, Any, Union

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', force=True)
logger = logging.getLogger(__name__)

s3_client = boto3.client('s3')

def merge_references(refs1: Dict, refs2: Dict) -> Union[Dict, List[Dict]]:
    """Merge reference objects or convert to array if they're different"""
    if not refs1:
        return refs2
    if not refs2:
        return refs1
        
    if isinstance(refs1, list):
        refs_list = refs1
    else:
        refs_list = [refs1]
        
    if isinstance(refs2, list):
        refs_list.extend(refs2)
    else:
        refs_list.append(refs2)
        
    # Deduplicate references based on all fields
    unique_refs = []
    seen = set()
    for ref in refs_list:
        ref_tuple = tuple(sorted(ref.items()))
        if ref_tuple not in seen:
            seen.add(ref_tuple)
            unique_refs.append(ref)
            
    return unique_refs if len(unique_refs) > 1 else unique_refs[0]

def consolidate_dynamic_data(items: List[Dict]) -> List[Dict]:
    """
    Consolidate any list of dictionaries with References, handling dynamic keys and nested structures
    """
    consolidated = {}
    
    for item in items:
        if not item:
            continue
            
        for key, value in item.items():
            # Handle case where value is a list
            if isinstance(value, list):
                if key not in consolidated:
                    consolidated[key] = []
                
                for v in value:
                    if isinstance(v, dict):
                        # Check if this dict has a References field
                        if 'References' in v:
                            # Try to merge with existing entries
                            merged = False
                            for existing in consolidated[key]:
                                if all(k == 'References' or existing.get(k) == v.get(k) 
                                     for k in set(existing.keys()) | set(v.keys()) - {'References'}):
                                    existing['References'] = merge_references(
                                        existing.get('References'),
                                        v.get('References')
                                    )
                                    merged = True
                                    break
                            if not merged:
                                consolidated[key].append(deepcopy(v))
                        else:
                            # Regular dict without References
                            if v not in consolidated[key]:
                                consolidated[key].append(deepcopy(v))
                    else:
                        # Simple value
                        if v not in consolidated[key]:
                            consolidated[key].append(v)
            
            # Handle case where value is a dict
            elif isinstance(value, dict):
                if 'References' in value:
                    if key not in consolidated:
                        consolidated[key] = deepcopy(value)
                    else:
                        # Merge if everything except References matches
                        if all(k == 'References' or consolidated[key].get(k) == value.get(k) 
                             for k in set(consolidated[key].keys()) | set(value.keys()) - {'References'}):
                            consolidated[key]['References'] = merge_references(
                                consolidated[key].get('References'),
                                value.get('References')
                            )
                else:
                    # Regular dict without References
                    if key not in consolidated:
                        consolidated[key] = deepcopy(value)
                    elif consolidated[key] != value:
                        # If values differ, convert to list
                        if not isinstance(consolidated[key], list):
                            consolidated[key] = [consolidated[key]]
                        if value not in consolidated[key]:
                            consolidated[key].append(value)
            
            # Handle simple values
            else:
                if key not in consolidated:
                    consolidated[key] = value
                elif consolidated[key] != value:
                    if not isinstance(consolidated[key], list):
                        consolidated[key] = [consolidated[key]]
                    if value not in consolidated[key]:
                        consolidated[key].append(value)
    
    # Convert consolidated dict to list format if needed
    if not consolidated:
        return []
    
    return [consolidated]

def merge_codes(codes: List[Dict]) -> List[Dict]:
    """Merge codes (Rx, ICD10CM, CPT) handling dynamic structures"""
    consolidated = {}
    
    for code_entry in codes:
        for code, details in code_entry.items():
            if code not in consolidated:
                consolidated[code] = deepcopy(details)
            else:
                # Merge all fields, treating References specially
                for field, value in details.items():
                    if field == 'References':
                        consolidated[code]['References'] = merge_references(
                            consolidated[code].get('References'),
                            value
                        )
                    elif field not in consolidated[code]:
                        consolidated[code][field] = value
                    elif consolidated[code][field] != value:
                        if not isinstance(consolidated[code][field], list):
                            consolidated[code][field] = [consolidated[code][field]]
                        if value not in consolidated[code][field]:
                            consolidated[code][field].append(value)
                        
    return [{k: v} for k, v in consolidated.items()]

def consolidate_procedures(procedures: List[Dict]) -> List[Dict]:
    """Special handling for ProceduresOrFindings, particularly KeywordsOrFindings"""
    consolidated = {}
    
    for proc in procedures:
        for key, values in proc.items():
            if key == 'References':
                continue
                
            if key not in consolidated:
                consolidated[key] = []
            
            # Handle array of findings
            if isinstance(values, list):
                for value in values:
                    # If value is dict with References
                    if isinstance(value, dict):
                        for finding_text, finding_details in value.items():
                            finding_entry = None
                            # Check if this finding already exists
                            for existing in consolidated[key]:
                                if isinstance(existing, dict) and finding_text in existing:
                                    finding_entry = existing
                                    break
                                elif not isinstance(existing, dict) and existing == finding_text:
                                    # Convert string to dict with References
                                    consolidated[key].remove(existing)
                                    finding_entry = {finding_text: {'References': proc.get('References', [])}}
                                    consolidated[key].append(finding_entry)
                                    break
                            
                            if finding_entry:
                                if isinstance(finding_details, dict) and 'References' in finding_details:
                                    existing_refs = finding_entry[finding_text].get('References', [])
                                    finding_entry[finding_text]['References'] = merge_references(
                                        existing_refs,
                                        finding_details['References']
                                    )
                            else:
                                consolidated[key].append({finding_text: finding_details})
                    # If value is string
                    else:
                        # Check if this string already exists
                        exists = False
                        for existing in consolidated[key]:
                            if isinstance(existing, dict) and value in existing:
                                exists = True
                                break
                            elif existing == value:
                                exists = True
                                break
                        
                        if not exists:
                            if proc.get('References'):
                                consolidated[key].append({value: {'References': proc['References']}})
                            else:
                                consolidated[key].append(value)
    
    return [consolidated] if consolidated else []

def transform_visit_data(data: Dict) -> Dict:
    """Transform visit data into more compact format"""
    transformed = deepcopy(data)
    
    # Special handling for ProceduresOrFindings
    if 'ProceduresOrFindings' in transformed:
        transformed['ProceduresOrFindings'] = consolidate_procedures(
            transformed['ProceduresOrFindings']
        )
    
    # Transform OtherInformation
    if 'OtherInformation' in transformed:
        transformed['OtherInformation'] = consolidate_dynamic_data(
            transformed['OtherInformation']
        )
    
    # Transform Codes
    if 'Codes' in transformed:
        codes = transformed['Codes']
        for code_type in codes:
            if isinstance(codes[code_type], list):
                codes[code_type] = merge_codes(codes[code_type])
    
    return transformed

def process_file(bucket_name: str, input_key: str, output_key: str):
    """Process a single JSON file from S3"""
    try:
        # Read input file
        response = s3_client.get_object(Bucket=bucket_name, Key=input_key)
        data = json.loads(response['Body'].read().decode('utf-8'))
        
        # Transform data
        transformed_data = transform_visit_data(data)
        
        # Write transformed data
        s3_client.put_object(
            Bucket=bucket_name,
            Key=output_key,
            Body=json.dumps(transformed_data, indent=2)
        )
        logger.info(f"Successfully transformed and saved: {output_key}")
        
    except Exception as e:
        logger.error(f"Error processing file {input_key}: {str(e)}")
        raise

def main(input_json: str):
    try:
        data = json.loads(input_json)
        bucket_name = data['bucket']
        user_id = data['user_id']
        case_id = data['case_id']
        input_prefix = data['input_prefix']
        output_prefix = data['output_prefix']
        
        # List all JSON files in input prefix
        paginator = s3_client.get_paginator('list_objects_v2')
        for page in paginator.paginate(Bucket=bucket_name, Prefix=input_prefix):
            if 'Contents' in page:
                for obj in page['Contents']:
                    if obj['Key'].endswith('.json'):
                        input_key = obj['Key']
                        output_key = output_prefix + input_key.split('/')[-1]
                        process_file(bucket_name, input_key, output_key)
        
        logger.info("Transformation completed successfully")
        
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        raise

if __name__ == "__main__":
    if len(sys.argv) != 2:
        logger.error("Usage: python transform_json.py '<json_input>'")
        sys.exit(1)
    
    json_input = sys.argv[1]
    main(json_input)