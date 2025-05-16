from main import app as application

# This file enables gunicorn to run the FastAPI application through a WSGI interface

# For working with gunicorn, make this application available
app = application