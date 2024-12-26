"""
Documentation for the /index endpoint
"""
from flask_restx import Resource

def register_root_endpoint(ns, models):
    """Register the root endpoint with the given namespace"""

    @ns.route('/index')
    class Root(Resource):
        """Root endpoint that redirects to Swagger documentation"""

        @ns.doc(description='Redirect to Swagger UI documentation')
        @ns.response(302, 'Redirect to Swagger UI', models['redirect_response'])
        def get(self):
            """
            Redirect to API documentation
            
            This endpoint redirects users to the Swagger UI documentation interface
            where they can explore and test the API endpoints.
            """
            pass  # Implementation is in app.py 