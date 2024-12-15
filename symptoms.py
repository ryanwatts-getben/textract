#!/usr/bin/env python3
import sys
import requests
import urllib.parse

def search_conditions(term, 
                      base_url="https://clinicaltables.nlm.nih.gov/api/conditions/v3/search",
                      df="consumer_name,primary_name",
                      ef="icd10cm_codes,term_icd9_code,term_icd9_text,synonyms",
                      sf="consumer_name,primary_name,word_synonyms,synonyms,term_icd9_code,term_icd9_text",
                      count=7,
                      offset=0):
    """
    Searches the Clinical Tables Conditions API for a given term.

    Parameters:
    -----------
    term : str
        The search term (partial word allowed) for the condition.
    base_url : str
        The base URL of the conditions API.
    df : str
        Fields to display as strings in results.
    ef : str
        Extra fields to retrieve (as JSON).
    sf : str
        Fields to search.
    count : int
        Number of results to return (page size).
    offset : int
        Starting offset for pagination.

    Returns:
    --------
    dict
        A dictionary containing parsed result data.
    """
    params = {
        "terms": term,
        "df": df,
        "ef": ef,
        "sf": sf,
        "count": count,
        "offset": offset
    }

    # Build query URL
    query_url = f"{base_url}?{urllib.parse.urlencode(params)}"

    # Make the request
    response = requests.get(query_url)
    response.raise_for_status()  # raise an error if the request failed
    data = response.json()

    # The returned JSON format is:
    # [
    #   total_count,
    #   [list_of_codes],
    #   {field_name: [values_for_each_item], ...},  # this corresponds to 'ef' fields
    #   [[df_field_values_for_item_1],[df_field_values_for_item_2], ...],
    #   [list_of_code_systems] (if available)
    # ]

    total_count = data[0]
    codes = data[1]
    extra_fields = data[2] if len(data) > 2 else {}
    display_fields = data[3] if len(data) > 3 else []
    code_systems = data[4] if len(data) > 4 else []

    results = []
    for i, code in enumerate(codes):
        # Match up the df fields
        df_values = display_fields[i] if i < len(display_fields) else []
        
        # Extract fields from ef
        item = {
            "code": code,
            "display_fields": df_values,
        }
        for field_name, field_values in extra_fields.items():
            item[field_name] = field_values[i] if i < len(field_values) else None
        
        # Optionally include code_system if present
        if code_systems:
            item["code_system"] = code_systems[i] if i < len(code_systems) else None
        
        results.append(item)

    return {
        "total_count": total_count,
        "results": results
    }

if __name__ == "__main__":
    # Get the search term either from command-line arguments or default to "lung cancer"
    if len(sys.argv) > 1:
        search_term = " ".join(sys.argv[1:])
    else:
        search_term = "lung cancer"

    result_data = search_conditions(search_term)

    # Print out the results
    print(f"Total matches found: {result_data['total_count']}")
    for idx, res in enumerate(result_data["results"]):
        print(f"\nResult {idx+1}:")
        print(f" Code: {res['code']}")
        if 'display_fields' in res:
            # The df fields we requested were consumer_name,primary_name
            # So display_fields will be something like: ["Consumer Name", "Primary Name"]
            if len(res['display_fields']) >= 2:
                consumer_name = res['display_fields'][0]
                primary_name = res['display_fields'][1]
                print(f" Consumer Name: {consumer_name}")
                print(f" Primary Name: {primary_name}")
        # Print extra fields from ef
        if 'icd10cm_codes' in res and res['icd10cm_codes']:
            print(f" ICD-10-CM Codes: {res['icd10cm_codes']}")
        if 'term_icd9_code' in res and res['term_icd9_code']:
            print(f" ICD-9-CM Code: {res['term_icd9_code']}")
        if 'term_icd9_text' in res and res['term_icd9_text']:
            print(f" ICD-9-CM Text: {res['term_icd9_text']}")
        if 'synonyms' in res and res['synonyms']:
            print(f" Synonyms: {res['synonyms']}")
