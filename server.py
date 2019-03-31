from flask import Flask, url_for, send_from_directory, request
import logging, os
import datetime
from werkzeug import secure_filename
from get_color_grids import process_image
from salila import salila_nn
import time

app = Flask(__name__)
file_handler = logging.FileHandler('server.log')
app.logger.addHandler(file_handler)
app.logger.setLevel(logging.INFO)

PROJECT_HOME = os.path.dirname(os.path.realpath(__file__))
UPLOAD_FOLDER = '{}/uploads/'.format(PROJECT_HOME)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

lst = ["test"]
del lst[0]

def create_new_folder(local_dir):
    newpath = local_dir
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    return newpath

@app.route('/csv', methods = ['GET'])
def csv():
    str = "time,file,test,r,g,b,label"
    for line in lst:
        str += '\n'
        str += line
    return str

@app.route('/<path:path>')
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
    app.logger.info(PROJECT_HOME)
    if request.method == 'POST' and request.files['image']:
        app.logger.info(app.config['UPLOAD_FOLDER'])
        img = request.files['image']
        img_name = secure_filename(img.filename)
        create_new_folder(app.config['UPLOAD_FOLDER'])
        saved_path = os.path.join(app.config['UPLOAD_FOLDER'], img_name)
        app.logger.info("saving {}".format(saved_path))
        img.save(saved_path)
        if True :  # try:
          #app.logger.info('img_name:%s' % (img_name))
          test_name = img_name[img_name.rfind('_')+1:].replace('.jpg','')
          app.logger.info(' test_name:%s , img_name:%s , saved_name:%s' % (test_name, img_name, saved_path))
          r,g,b,out_file = process_image(saved_path, test_name)
          app.logger.info('outfile: %s' % out_file)
          y_pred, x_instance = salila_nn.salila_ml(out_file,r,g,b)
          app.logger.info('x_instance: %s' % x_instance)
          app.logger.info('y_pred: %s' % y_pred)
          tx = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
          #f=open('tranlog.csv', 'a')
          #f.write(datetime.datetime.now() + "," + img_name + "," + test_name + "," + y_pred)
          #f.close()

          lst.insert(0, ",".join([tx, img_name, test_name, str(r), str(g) ,str(b) , y_pred[0]  ]) )

          with open("tranlog.csv", "a") as myfile:
              myfile.write(",".join([tx, img_name, test_name, str(r), str(g) ,str(b) , y_pred[0]  ])); myfile.write("\n")
        else : #except:
          pass

        return send_from_directory(app.config['UPLOAD_FOLDER'],img_name, as_attachment=True)
    else:
        return "Where is the image?"

if __name__ == '__main__':
    f = open('tranlog.csv', 'a')
    f.close()
    f = open("tranlog.csv", "r")
    for line in f:
        lst.insert(0, line)
    f.close()
    app.run(host='0.0.0.0', debug=False)
