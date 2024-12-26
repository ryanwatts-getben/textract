"""
Documentation for the /generate-multiple-disease-definitions endpoint
"""
from flask_restx import Resource

def register_disease_endpoint(ns, models):
    """Register the disease definitions endpoint with the given namespace"""

    @ns.route('/generate-multiple-disease-definitions')
    class DiseaseDefinitions(Resource):
        """Endpoint for generating multiple disease definitions"""

        @ns.doc(description='Generate definitions for multiple diseases based on provided criteria')
        @ns.expect(models['disease_request'])
        @ns.response(200, 'Success', models['disease_response'])
        @ns.response(400, 'Invalid request')
        @ns.response(500, 'Internal server error')
        def post(self):
            """
            Generate multiple disease definitions
            
            This endpoint accepts a list of diseases with their associated criteria
            and generates comprehensive definitions for each disease. The definitions
            are based on the provided symptoms, lab results, procedures, risk factors,
            treatments, and complications.
            
            Example request:
            {
                "diseases": [
                    {
                        "name": "Traumatic Brain Injury",
                        "symptoms": [
                            "Loss of consciousness",
                            "Headache",
                            "Confusion"
                        ],
                        "lab_results": [
                            "CT scan showing brain hemorrhage",
                            "Elevated intracranial pressure"
                        ],
                        "procedures": [
                            "Neurological examination",
                            "Glasgow Coma Scale assessment"
                        ],
                        "risk_factors": [
                            "Contact sports",
                            "Motor vehicle accidents"
                        ],
                        "treatments": [
                            "Rest",
                            "Pain management",
                            "Rehabilitation therapy"
                        ],
                        "complications": [
                            "Post-concussion syndrome",
                            "Cognitive impairment"
                        ]
                    }
                ]
            }
            """
            pass  # Implementation is in app.py 