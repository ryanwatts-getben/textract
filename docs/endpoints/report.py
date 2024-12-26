"""
Documentation for the /api/disease-scanner/reports/generate-report endpoint
"""
from flask_restx import Resource

def register_report_endpoint(ns, models):
    """Register the report generation endpoint with the given namespace"""

    @ns.route('/api/disease-scanner/reports/generate-report')
    class ReportGenerator(Resource):
        """Endpoint for generating disease scan reports"""

        @ns.doc(description='Generate a comprehensive report from disease scanning results')
        @ns.expect(models['report_request'])
        @ns.response(200, 'Success', models['report_response'])
        @ns.response(400, 'Invalid request')
        @ns.response(500, 'Internal server error')
        def post(self):
            """
            Generate disease scan report
            
            This endpoint accepts disease scanning results and generates a comprehensive
            report detailing the findings, confidence scores, and supporting evidence
            for each identified disease.
            
            Example request:
            {
                "userId": "user123",
                "projectId": "project456",
                "scanResults": [
                    {
                        "disease_name": "Traumatic Brain Injury",
                        "confidence_score": 0.85,
                        "findings": [
                            {
                                "text": "Patient experienced loss of consciousness",
                                "confidence": 0.92,
                                "page": 1,
                                "document_name": "ER_Report_2023.pdf",
                                "source_type": "medical_record"
                            }
                        ],
                        "category_scores": {
                            "symptoms": 0.88,
                            "lab_results": 0.82,
                            "procedures": 0.85
                        }
                    }
                ],
                "format": "PDF"
            }
            """
            pass  # Implementation is in app.py 