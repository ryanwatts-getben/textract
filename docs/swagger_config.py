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

# Import and initialize models
from .models.common_models import init_common_models
from .models.root_models import init_root_models
from .models.scan_models import init_scan_models
from .models.scrape_models import init_scrape_models
from .models.index_models import init_index_models
from .models.query_models import init_query_models
from .models.disease_models import init_disease_models
from .models.report_models import init_report_models

common_models = init_common_models(api)
root_models = init_root_models(api)
scan_models = init_scan_models(api)
scrape_models = init_scrape_models(api)
index_models = init_index_models(api)
query_models = init_query_models(api)
disease_models = init_disease_models(api)
report_models = init_report_models(api)

# Combine models with common models
root_models.update(common_models)
scan_models.update(common_models)
scrape_models.update(common_models)
index_models.update(common_models)
query_models.update(common_models)
disease_models.update(common_models)
report_models.update(common_models)

# Import and register endpoints
from .endpoints.root import register_root_endpoint
from .endpoints.beginScan import register_beginScan_endpoint
from .endpoints.scrape import register_scrape_endpoint
from .endpoints.index import register_index_endpoint
from .endpoints.query import register_query_endpoint
from .endpoints.disease import register_disease_endpoint
from .endpoints.report import register_report_endpoint

register_root_endpoint(root_ns, root_models)
register_beginScan_endpoint(scan_ns, scan_models)
register_scrape_endpoint(scrape_ns, scrape_models)
register_index_endpoint(index_ns, index_models)
register_query_endpoint(query_ns, query_models)
register_disease_endpoint(disease_ns, disease_models)
register_report_endpoint(report_ns, report_models) 