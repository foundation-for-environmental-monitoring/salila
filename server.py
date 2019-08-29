from flask import Flask, url_for, send_from_directory, request, abort
import logging, os
import datetime
from werkzeug import secure_filename
import time
import boto3
import json
import redis
import decimal
import hashlib
from rq import Queue

from salila.analyze import analyse_task
from salila.extract_colors import process_image
from salila import predict_result

app = Flask(__name__)
file_handler = logging.FileHandler('log/server.log')
app.logger.addHandler(file_handler)
app.logger.setLevel(logging.INFO)

PROJECT_HOME = os.path.dirname(os.path.realpath(__file__))
UPLOAD_FOLDER = '{}/uploads/'.format(PROJECT_HOME)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
LOG_FOLDER = '{}/log/'.format(PROJECT_HOME)

r = redis.Redis()
q = Queue(connection=r)

# Helper class to convert a DynamoDB item to JSON.
class DecimalEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, decimal.Decimal):
            if o % 1 > 0:
                return float(o)
            else:
                return int(o)
        return super(DecimalEncoder, self).default(o)

def create_new_folder(local_dir):
    newpath = local_dir
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    return newpath

@app.route('/result', methods = ['GET'])
def result():
    dynamodb = boto3.resource('dynamodb', region_name='ap-south-1')
    table = dynamodb.Table('test_details')

    id  = request.args.get('id', "0")
    response = table.get_item(
       Key = {
            'TestRunId': id
        }
    )

    try:
        return json.dumps(response['Item'], indent=4, cls=DecimalEncoder)
    except:
        return f'{{ "TestRunId": "{id}", "result": "-", "message": "Error", "title": "Fluoride" }}'

def create_table():
    # Get the service resource.
    dynamodb = boto3.resource('dynamodb', region_name='ap-south-1')

#@app.route('/<path:path>')
def get_resource(path):  # pragma: no cover
    mimetypes = {
        ".css": "text/css",
        ".html": "text/html",
        ".js": "application/javascript",
    }
    complete_path = os.path.join(root_dir(), path)
    ext = os.path.splitext(path)[1]
    mimetype = mimetypes.get(ext, "text/html")
    content = get_file(complete_path)
    return Response(content, mimetype=mimetype)

@app.route('/', methods = ['POST'])
def api_root():
    if request.method == 'POST' and request.files['image']:
        img = request.files['image']
        test_id = request.form['testId']
        version_code = request.form['versionCode']
        sdk_version = request.form['sdkVersion']
        device_model = request.form['deviceModel']
        md5 = request.form['md5']
        img_name = secure_filename(img.filename)
        create_new_folder(app.config['UPLOAD_FOLDER'])
        saved_path = os.path.join(app.config['UPLOAD_FOLDER'], img_name)
        app.logger.info("saving {}".format(img_name))
        img.save(saved_path)

        hasher = hashlib.md5()
        with open(saved_path, 'rb') as afile:
            buf = afile.read()
            hasher.update(buf)

        if md5 == hasher.hexdigest():
            tx = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            test_name = img_name[img_name.rfind('_')+1:].replace('.jpg','')

            dynamodb = boto3.resource('dynamodb', region_name='ap-south-1')
            table = dynamodb.Table('test_details')
            message =  'Analyzing'
            response = table.put_item(
            Item = {
                    'TestRunId': test_id,
                    'date': int(time.time_ns() * 0.000001),
                    'image': img_name,
                    'title': test_name,
                    'message': message,
                    'appVersion': version_code,
                    'sdk': sdk_version,
                    'model': device_model
                }
            )

            job = q.enqueue(analyse_task, img_name, saved_path, test_id)
            return f"task {job.id} at {job.enqueued_at}"

        return abort(400)
    else:
        return "Where is the image?"

if __name__ == '__main__':
    create_table()

    create_new_folder(LOG_FOLDER)
    if not os.path.isfile('results.csv'):
        f = open('log/results.csv', 'a')
        f.close()
    
    app.run(host='0.0.0.0', debug=False)
