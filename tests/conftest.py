import os
import tempfile
# Set before importing app modules, including when a single test file is selected.
os.environ['APP_DATA_DIR']=tempfile.mkdtemp(prefix='outbreak-tests-')
os.environ.pop('APP_API_TOKEN',None)
