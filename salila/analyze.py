from flask import Flask, url_for, send_from_directory, request
import logging, os
import datetime
from werkzeug import secure_filename
import boto3
import time

from salila.extract_colors import process_image
from salila import predict_result

app = Flask(__name__)

def analyse_task(img_name, saved_path, test_id):
        app.logger.info('img_name:%s' % (img_name))
        test_name = img_name[img_name.rfind('_')+1:].replace('.jpg','')
        app.logger.info(' test_name:%s , img_name:%s , saved_name:%s' % (test_name, img_name, saved_path))
        r,g,b,out_file = process_image(saved_path, test_name)
        result = '-'
        message = '-'

        tx = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

        if out_file != '':
          app.logger.info('outfile: %s' % out_file)
          y_pred, x_instance = predict_result.salila_ml(out_file,r,g,b)
          app.logger.info('x_instance: %s' % x_instance)
          app.logger.info('y_pred: %s' % y_pred)

          result = y_pred[0].replace(test_name, '')

          with open("log/results.csv", "a") as myfile:
            myfile.write(",".join([tx, img_name, test_name, str(r), str(g) ,str(b) , y_pred[0]  ])); myfile.write("\n")
        else:
          message = 'Image error'

        dynamodb = boto3.resource('dynamodb', region_name='ap-south-1')
        table = dynamodb.Table('test_details')
        response = table.update_item(
            Key= {
                'TestRunId': test_id
            },
            UpdateExpression='set #result = :r, message = :m',
            ExpressionAttributeValues={
                ':r': result,
                ':m': message
            },
            ExpressionAttributeNames={ '#result': "result" }
        )
    