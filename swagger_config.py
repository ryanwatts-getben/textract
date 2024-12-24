from flask_restx import Api, fields  # type: ignore

# Initialize API
api = Api(
    title='MedChron API',
    version='1.0',
    description='Medical Document Processing and Disease Information API',
    doc='/docs',
    ordered=True,
    validate=True
)

# Create namespaces
scrape_ns = api.namespace('scrape', description='Web scraping operations')
scan_ns = api.namespace('scan', description='Document scanning and analysis operations')

# Error Response Model
error_response = api.model('ErrorResponse', {
    'error': fields.String(description='Error message')
})

# Symptom Model
symptom_model = api.model('Symptom', {
    'id': fields.String(description='Symptom ID'),
    'name': fields.String(description='Symptom name'),
    'commonality': fields.String(description='How common the symptom is'),
    'severity': fields.String(description='Severity of the symptom')
})

# Lab Result Model
lab_result_model = api.model('LabResult', {
    'id': fields.String(description='Lab result ID'),
    'name': fields.String(description='Lab test name'),
    'normalRange': fields.String(description='Normal range for the test'),
    'unit': fields.String(description='Unit of measurement')
})

# Diagnostic Procedure Model
diagnostic_procedure_model = api.model('DiagnosticProcedure', {
    'id': fields.String(description='Procedure ID'),
    'name': fields.String(description='Procedure name'),
    'accuracy': fields.Float(description='Accuracy rate of the procedure'),
    'invasiveness': fields.String(description='Level of invasiveness')
})

# Risk Factor Model
risk_factor_model = api.model('RiskFactor', {
    'id': fields.String(description='Risk factor ID'),
    'name': fields.String(description='Risk factor name'),
    'impact': fields.String(description='Impact level of the risk factor')
})

# Disease Model
disease_model = api.model('Disease', {
    'id': fields.String(description='Disease ID'),
    'name': fields.String(description='Disease name'),
    'symptoms': fields.List(fields.Nested(symptom_model)),
    'labResults': fields.List(fields.Nested(lab_result_model)),
    'diagnosticProcedures': fields.List(fields.Nested(diagnostic_procedure_model)),
    'riskFactors': fields.List(fields.Nested(risk_factor_model)),
    'scoringModel': fields.Raw(description='Scoring model configuration')
})

# Mass Tort Model
mass_tort_model = api.model('MassTort', {
    'id': fields.String(description='Mass tort ID'),
    'officialName': fields.String(description='Official name of the mass tort'),
    'diseases': fields.List(fields.Nested(disease_model))
})

# Scan Analysis Request Model
scan_request = api.model('ScanRequest', {
    'userId': fields.String(required=True, description='User ID'),
    'projectId': fields.String(required=True, description='Project ID'),
    'massTorts': fields.List(fields.Nested(mass_tort_model), required=True,
        description='List of mass torts with their associated diseases')
})

# Scan Analysis Response Model
scan_response = api.model('ScanResponse', {
    'status': fields.String(description='Status of the scan operation'),
    'message': fields.String(description='Status message'),
    'results': fields.Raw(description='Scan analysis results')
})

# Scrape Request/Response Models
scrape_request = api.model('ScrapeRequest', {
    'diseaseUrl': fields.Raw(required=True, 
        description='URL or array of [URL, name] pairs. Can be either a single URL string or an array of [URL, name] pairs.',
        example=[
            ["https://medlineplus.gov/genetics/condition/non-alcoholic-fatty-liver-disease/", 
             "Non-Alcoholic Fatty Liver Disease"]
        ])
})

scrape_response = api.model('ScrapeResponse', {
    'title': fields.String(description='Disease title', 
        example='Non-Alcoholic Fatty Liver Disease'),
    'also_called': fields.String(description='Alternative names for the disease',
        example='NAFLD, Fatty Liver Disease'),
    'summary': fields.String(description='Disease summary',
        example='Non-alcoholic fatty liver disease is a condition in which excess fat builds up in liver cells...'),
    'symptoms': fields.List(fields.String, description='List of symptoms',
        example=[
            'Fatigue',
            'Pain in upper right abdomen',
            'Enlarged liver',
            'Jaundice'
        ]),
    'lab_results': fields.List(fields.String, description='List of laboratory tests',
        example=[
            'Liver function tests',
            'Complete blood count',
            'Lipid profile'
        ]),
    'diagnostic_procedures': fields.List(fields.String, description='List of diagnostic procedures',
        example=[
            'Ultrasound',
            'CT scan',
            'Liver biopsy'
        ]),
    'treatments': fields.List(fields.String, description='List of treatments',
        example=[
            'Weight loss',
            'Dietary changes',
            'Exercise program',
            'Medications'
        ]),
    'risk_factors': fields.List(fields.String, description='List of risk factors',
        example=[
            'Obesity',
            'Type 2 diabetes',
            'High cholesterol'
        ]),
    'complications': fields.List(fields.String, description='List of potential complications',
        example=[
            'Cirrhosis',
            'Liver cancer',
            'Liver failure'
        ]),
    'prevention': fields.List(fields.String, description='List of prevention methods',
        example=[
            'Maintain healthy weight',
            'Exercise regularly',
            'Eat a balanced diet'
        ]),
    'when_to_see_doctor': fields.List(fields.String, description='When to seek medical attention',
        example=[
            'Severe abdominal pain',
            'Yellowing of skin or eyes',
            'Dark urine color'
        ])
})

scrape_multiple_response = api.model('ScrapeMultipleResponse', {
    '*': fields.Nested(scrape_response, description='Response for each disease, keyed by disease name')
}) 