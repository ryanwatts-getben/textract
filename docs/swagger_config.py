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
    prefix='/api/v1',
    # Swagger UI configuration
    swagger_ui_bundle_js='https://unpkg.com/swagger-ui-dist@5.9.0/swagger-ui-bundle.js',
    swagger_ui_css='https://unpkg.com/swagger-ui-dist@5.9.0/swagger-ui.css',
    swagger_ui_config={
        'deepLinking': True,
        'displayRequestDuration': True,
        'filter': True,
        'showExtensions': True,
        'showCommonExtensions': True,
        'defaultModelsExpandDepth': -1,
        'defaultModelExpandDepth': 5,
        'defaultModelRendering': 'model',
        'docExpansion': 'full',
        'tryItOutEnabled': True,
        'displayOperationId': True,
                'customCss': '''
            .swagger-ui .copy-to-clipboard {
                display: inline-flex !important;
                opacity: 1 !important;
                visibility: visible !important;
                background-color: #f0f0f0 !important;
                position: relative !important;
                cursor: pointer;
                border: 1px solid #ddd;
                border-radius: 4px;
                padding: 4px 8px;
                margin-left: 8px;
            }
        ''',
        'customJs': '/static/js/swagger-custom.js',
        'maxDisplayedTags': None,
        'showExtensions': True,
        'showCommonExtensions': True
    }
)

# Create namespaces
scrape_ns = api.namespace('scrape', description='Web scraping operations')
scan_ns = api.namespace('scan', description='Document scanning and analysis operations')

# Import and initialize models
from .models.common_models import init_common_models
from .models.scan_models import init_scan_models
from .models.scrape_models import init_scrape_models

common_models = init_common_models(api)
scan_models = init_scan_models(api)
scrape_models = init_scrape_models(api)

# Combine models with common models
scan_models.update(common_models)
scrape_models.update(common_models)

# Import and register endpoints
from .endpoints.beginScan import register_beginScan_endpoint
from .endpoints.scrape import register_scrape_endpoint

register_beginScan_endpoint(scan_ns, scan_models)
register_scrape_endpoint(scrape_ns, scrape_models) 