# textract
```bash
brew cask install xquartz
brew install poppler antiword unrtf tesseract swig
pip install textract
```

I had to use this because the above failed:
`pip install textract==1.6.3`

I also did this
`pip install chardet pdfminer.six docx2txt extract-msg==0.28.7 SpeechRecognition xlrd EbookLib`

and 

`pip install pypdf`

get the BioBERT-mnli-snli-scinli-scitail-mednli-stsb folder in root

# Salesforce Client Integration

This section describes how to use the Salesforce Client Integration features.

## Fetching Matter Context from Salesforce and Creating Projects

### POST /nulaw

You can fetch matter context from Salesforce and create a new project by sending a POST request to the `/nulaw` endpoint.

```bash
curl -X POST "http://localhost:5000/nulaw" \
-H "Content-Type: application/json" \
-d '{
  "matter_id": "a0OUR000004DwOr2AK",
  "sf_path": "C:\\path\\to\\sf.cmd",
  "download_files": true
}'
```

Parameters:
- `matter_id` (required): The Salesforce Matter ID to retrieve context for
- `sf_path` (optional): Path to the Salesforce CLI executable
- `download_files` (optional, default: true): Whether to download files from SharePoint

### GET /nulaw/{matter_id}

You can also fetch matter context and create a project using a GET request with the matter ID in the URL:

```bash
curl -X GET "http://localhost:5000/nulaw/a0OUR000004DwOr2AK?sf_path=C:\\path\\to\\sf.cmd&download_files=false"
```

Query parameters:
- `sf_path` (optional): Path to the Salesforce CLI executable
- `download_files` (optional, default: true): Whether to download files from SharePoint

## Project Creation Process

When calling either of the `/nulaw` endpoints, the system will:

1. Fetch the matter context from Salesforce
2. Extract key information like client name, incident date, and medical billing amounts
3. Automatically create a new project by calling the project creation endpoint
4. Return the original matter context along with the project creation result

The created project will include all necessary fields from the Salesforce matter record, including:
- Client name (from `Client_Full_Name__c`)
- Incident date (from `Accident_Date_Time__c`, with time portion removed)
- Medical bills total (from `Treatments_Total_Billed_Amount__c`)
- Case type information (from `nu_law__Case_Type__c` and `nu_law__Sub_Case_Type__c`)
- Additional context from the matter record

### Controlling File Downloads

You can control whether files are downloaded from SharePoint by setting the `download_files` parameter:
- When set to `true` (default), files from SharePoint will be downloaded
- When set to `false`, no files will be downloaded from SharePoint

## Manual Project Creation

If you prefer to create projects manually without using the API, you can use the `salesforce_create_new_client.py` script directly:

```bash
python salesforce_create_new_client.py --matter-id a0OUR000004DwOr2AK --sf-path "C:\\path\\to\\sf.cmd" --no-download-files
```

Options:
- `--matter-id` (required): The Salesforce Matter ID to create a project for
- `--sf-path` (optional): Path to the Salesforce CLI executable
- `--dry-run` (optional): Only display the payload without creating the project
- `--download-files` (optional, default: true): Download files from SharePoint
- `--no-download-files` (optional): Do not download files from SharePoint