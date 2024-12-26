"""
Documentation for the /beginScan endpoint
"""
from flask_restx import Resource

def register_beginScan_endpoint(scan_ns, models):
    """Register the beginScan endpoint with its documentation"""
    
    @scan_ns.route('/beginScan')
    class BeginScan(Resource):
        """Endpoint for scanning medical documents for disease evidence."""
        
        @scan_ns.doc('beginScan',
            description='Analyze medical documents for evidence of specified diseases within mass torts.')
        @scan_ns.expect(models['scan_request'], validate=True)
        @scan_ns.response(200, 'Success', models['beginScan_response'])
        @scan_ns.response(400, 'Validation Error', models['error_response'])
        @scan_ns.response(500, 'Internal Server Error', models['error_response'])
        def post(self):
            """
            Scan medical documents for disease evidence.
            
            This endpoint analyzes medical documents for evidence of specified diseases within mass torts.
            It performs detailed analysis of symptoms, lab results, procedures, and risk factors,
            providing confidence scores and relevant excerpts from the documents.
            
            The analysis includes:
            - Individual matching of symptoms, lab results, procedures, and risk factors
            - Confidence scoring based on medical context and terminology
            - Extraction of relevant text excerpts with confidence context
            - Category-wise and overall confidence scoring
            
            Example Usage:
            ```json
            {
              "userId": "4e6f9de5-5128-4ff4-980c-2429e31ec2ec",
              "projectId": "ea2fc6b9-7759-470f-9e48-93b364eee330",
              "massTorts": [
                {
                  "id": "02ccfcb0-1658-4242-a4c9-0b0d4de17e85",
                  "officialName": "Aqueous Film-Forming Foam",
                  "diseases": [
                    {
                      "id": "57bab3b6-41c6-4255-bcb0-e0fb242a12b9",
                      "name": "Kidney Cancer",
                      "symptoms": [
                        "Blood in urine",
                        "Lower back pain",
                        "Fatigue",
                        "Weight loss",
                        "Loss of appetite",
                        "Fever",
                        "High blood pressure",
                        "Anemia"
                      ],
                      "labResults": [
                        {
                          "name": "Complete Blood Count: 4.5-11.0 x10^9/L",
                          "range": "Not specified"
                        },
                        {
                          "name": "Creatinine: 0.7-1.3 mg/dL",
                          "range": "Not specified"
                        },
                        {
                          "name": "Calcium: 8.5-10.5 mg/dL",
                          "range": "Not specified"
                        }
                      ],
                      "diagnosticProcedures": [
                        "CT scan",
                        "MRI",
                        "Ultrasound",
                        "Biopsy",
                        "PET scan",
                        "Bone scan",
                        "Chest X-ray",
                        "Urinalysis"
                      ],
                      "riskFactors": [
                        "Smoking",
                        "Obesity",
                        "Hypertension",
                        "Family history",
                        "Von Hippel-Lindau disease",
                        "Advanced kidney disease",
                        "Workplace exposure to toxins"
                      ],
                      "scoringModel": {
                        "id": "e4d52e91-1114-4264-9dcf-71bec597c0b3",
                        "confidenceThreshold": 0.7
                      }
                    }
                  ]
                }
              ]
            }
            ```
            """
            pass  # Implementation is in scan.py
