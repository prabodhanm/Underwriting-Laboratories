# from flask import Blueprint, Flask
import json

#call_neiss_api
# spark_api = Flask(__name__)

# app.register_blueprint(spark_api)

# spark_api = Blueprint('spark_api', __name__)

# @spark_api.route('/prabodhan/')
def sayhello():
  return jsonify(
    status="Hi"
  )

# @spark_api.route('/ppe_cpsc_neiss/<key>/')
def call_neiss_api(key):
  print("Key = ", key)
  readConfig = {
    "EndPoint" : "https://neissapi.documents.azure.com:443/",
    "Masterkey" : "cvDEpUyy0jrAa4gsY36zfqiZ1ZuIsPcpADDpkAHoVmiYpQ416sfjyYpj7SM8rVil0qMpLmV4Bgv3hD2NGL8wTw==",
    "Database" : "neissdb",
    "Collection" : "neisscollection",
    "query_custom" : "select c.id,c.norm from c where c.id=% order by c.id" % key
  }

  neiss_data = spark.read.format("com.microsoft.azure.cosmosdb.spark").options(**readConfig).load()
  neiss_data.show()
  
  return jsonify(
            status="success",
            key=key
        )