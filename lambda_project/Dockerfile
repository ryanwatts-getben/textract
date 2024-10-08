FROM public.ecr.aws/lambda/python:3.9
RUN yum install -y zip
RUN pip install requests boto3 python-dotenv PyMuPDF

# Copy your Lambda function code into the container
COPY lambda_function.py .

# Set the CMD to your handler (for AWS Lambda to call)
CMD ["lambda_function.handler"]
