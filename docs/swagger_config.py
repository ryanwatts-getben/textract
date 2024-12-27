"""
Main Swagger configuration and API initialization
"""
from flask_restx import Api

# Initialize API with custom Swagger UI configuration
api = Api(
    title='MedChron API',
    version='1.0',
    description='Medical Document Processing and Disease Information API',
    doc='/swagger',
    ordered=True,
    validate=True,
    # Swagger UI configuration
    swagger_ui_bundle_js='https://unpkg.com/swagger-ui-dist@5.9.0/swagger-ui-bundle.js',
    swagger_ui_css='https://unpkg.com/swagger-ui-dist@5.9.0/swagger-ui.css',
    swagger_ui_config={
        'deepLinking': True,
        'displayRequestDuration': True,
        'filter': True,
        'showExtensions': True,
        'showCommonExtensions': True,
        'defaultModelsExpandDepth': 5,
        'defaultModelExpandDepth': 5,
        'defaultModelRendering': 'model',
        'docExpansion': 'full',
        'tryItOutEnabled': True,
        'displayOperationId': True,
        'maxDisplayedTags': None,
        'showExtensions': True,
        'showCommonExtensions': True
    }
)

# Create namespaces
root_ns = api.namespace('', description='Root endpoint operations')
scrape_ns = api.namespace('scrape', description='Web scraping operations')
scan_ns = api.namespace('', description='Document scanning and analysis operations')
index_ns = api.namespace('', description='Document indexing operations')
query_ns = api.namespace('', description='Document querying operations')
disease_ns = api.namespace('', description='Disease definition operations')
report_ns = api.namespace('api/disease-scanner/reports', description='Report generation operations')

# Import and register endpoints
from .endpoints.root import register_root_endpoint
from .endpoints.beginScan import register_beginScan_endpoint
from .endpoints.scrape import register_scrape_endpoint
from .endpoints.index import register_index_endpoint
from .endpoints.query import register_query_endpoint
from .endpoints.disease import register_disease_endpoint
from .endpoints.report import register_report_endpoint

register_root_endpoint(root_ns)
register_beginScan_endpoint(scan_ns)
register_scrape_endpoint(scrape_ns)
register_index_endpoint(index_ns)
register_query_endpoint(query_ns)
register_disease_endpoint(disease_ns)
register_report_endpoint(report_ns) 