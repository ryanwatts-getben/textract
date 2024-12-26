# API Documentation

This directory contains the Swagger/OpenAPI documentation for the MedChron API. The documentation is implemented using `flask-restx` and follows a separation of concerns pattern where all API documentation is centralized in this directory.

## Accessing the Documentation

### Local Development
When running the application locally, you can access the Swagger UI at:
```
http://localhost:5001/swagger
```

The documentation interface provides:
- Interactive API exploration
- Request/response schema details
- Example requests and responses
- Try-it-out functionality for testing endpoints

### Key Files

- `swagger_config.py`: Contains all Swagger models and route configurations
  - API models and schemas
  - Route documentation
  - Example request/response structures
  - Validation rules

## Documentation Structure

### Endpoints

1. **Scan Operations** (`/scan/*`)
   - `/beginScan` (POST): Analyze medical documents for disease evidence
     - Request: Mass tort and disease criteria
     - Response: Detailed analysis with confidence scores

2. **Scrape Operations** (`/scrape/*`)
   - `/scrape` (POST): Extract disease information from medical sources
     - Request: Disease URLs
     - Response: Structured disease information

### Models

1. **Scan Models**
   - `Match`: Individual finding matches with confidence
   - `CategoryScores`: Category-wise confidence scoring
   - `DiseaseAnalysis`: Complete disease analysis results
   - `MassTortAnalysis`: Mass tort level results
   - `BeginScanResponse`: Overall scan operation response

2. **Scrape Models**
   - `ScrapeRequest`: URL input structure
   - `ScrapeResponse`: Structured disease information
   - `ScrapeMultipleResponse`: Multiple disease responses

## Usage Tips

1. **Expanding Documentation**
   - All sections are expandable/collapsible
   - Click on model names to see detailed schemas
   - Use the "Try it out" button to test endpoints

2. **Authentication**
   - Some endpoints may require authentication
   - Use the "Authorize" button if available
   - API keys should be provided in headers

3. **Response Codes**
   - 200: Successful operation
   - 400: Validation error (check request format)
   - 500: Internal server error (check logs)

## Development Guidelines

1. **Adding New Endpoints**
   - Add models to `swagger_config.py`
   - Use descriptive names for models and fields
   - Include realistic example values
   - Document all possible response codes

2. **Documentation Best Practices**
   - Keep examples up-to-date
   - Include both success and error scenarios
   - Use clear, concise descriptions
   - Document any rate limits or constraints

## Troubleshooting

If you can't access the documentation:
1. Ensure the application is running
2. Check you're using the correct port
3. Look for any CORS issues in browser console
4. Verify the `/docs` endpoint is enabled in `app.py`

## Contributing

When adding or modifying endpoints:
1. Update the relevant models in `swagger_config.py`
2. Add comprehensive examples
3. Test the documentation UI
4. Update this README if needed 