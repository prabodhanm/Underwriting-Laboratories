from flask import Blueprint, jsonify
import json

# Spark ML
from source.spark.service_helpers import spark, es
from pyspark.ml.feature import BucketedRandomProjectionLSHModel, Tokenizer, StopWordsRemover, IDF, CountVectorizerModel, Normalizer
from pyspark.ml import Pipeline
import pyspark

spark_api = Blueprint('spark_api', __name__)
'''
neiss_vector = spark.read.load(
    "/mnt/spark-recommendation-models/neiss_model/neiss_tfidf.parquet"
).cache()
neiss_model = BucketedRandomProjectionLSHModel.load(
    "/mnt/spark-recommendation-models/neiss_model/neiss_distance.model")
neiss_vocab_model = CountVectorizerModel.load(
    "/mnt/spark-recommendation-models/neiss_model/neiss_vocab.model"
)'''

print("Running")


def load_spark_data(tfidf_path, model_path):
    data = spark.read.load(tfidf_path).cache()
    model = BucketedRandomProjectionLSHModel.load(model_path)
    return data, model


@spark_api.route('/test/')
def test_docker():
    print("inside")
    return "TEST DONE"


@spark_api.route('/ppe_fda_maude/<key>/')
def spark_approxnn(key):
    '''
    Function : Find similarity FDA
    Input    : key - mdr_key ID
    Output   : JSON - mdr_key Nearest Neighbors
    '''
    # Load computed spark dataframe and model

    try:
        # Load computed spark dataframe and model
        tfidf_path = "/mnt/spark-recommendation-models/fda_maude_model/data_vector.parquet"
        model_path = "/mnt/spark-recommendation-models/fda_maude_model/fda_distance.model"
        fda_vector, model = load_spark_data(tfidf_path, model_path)
        # Find 10 similarity
        spark_key = fda_vector.where(fda_vector.mdr_key == key).select(
            'norm').collect()[0]['norm']
        nn_spark = model.approxNearestNeighbors(
            fda_vector, spark_key, 21).select('mdr_key')
        nn_array = [str(row.mdr_key) for row in nn_spark.collect()]

        print(nn_array)

        return jsonify(
            status="success",
            key=key,
            data=nn_array[1:]
        )

    except IndexError as e:
        return jsonify(
            status="failed",
            key=key,
            data=e.args
        )

    except Exception as e:
        print("Unexpected error: check log file")
        print(e)
        return jsonify(
            status="failed",
            key=key,
            data="something went wrong"
        )


@spark_api.route('/refresh/')
def refresh_metadata():
    print("refreshing Test")
    spark.catalog.refreshTable("saferproducts_vector")
    return "Refreshing DONE"


@spark_api.route('/ppe_cpsc_neiss/<key>/')
def spark_neiss(key):
    '''
    Function : Find similarity NEISS
    Input    : key - _id ID
    Output   : JSON - _id Nearest Neighbors
    '''
    # Load computed spark dataframe and model

    try:
        def convert_na(x): return x if x is not None else ''
        # /mnt/spark-recommendation-models/neiss_model/neiss_code.json

        def create_narrative(item):
            with open('/mnt/spark-recommendation-models/neiss_model/neiss_code.json') as fp:
                code_map = json.load(fp)

            item['Fire_Involvement'] = code_map['fire'][item['Fire_Involvement']]
            item['Body_Part'] = code_map['bodypart'][item['Body_Part']]
            item['Disposition'] = code_map['disposition'][item['Disposition']]
            item['Product_1'] = code_map['product_code'][item['Product_1']]
            item['Product_2'] = code_map['product_code'][item['Product_2']
                                                         ] if item['Product_2']not in ['0', '', None] else None
            item['Diagnosis'] = code_map['diagnosis'][item['Diagnosis']]
            narrative = ' '.join([convert_na(i) for i in [item['Fire_Involvement'], item['Body_Part'], item['Disposition'],
                                                          item['Product_1'], item['Product_2'], item['Narrative_1'], item['Narrative_2'], item['Diagnosis']]])
            return spark.createDataFrame([(item['CPSC_Case_Number'], narrative.lower())], ["id", "narrative"])
            # Load computed spark dataframe and model
        neiss_vector = spark.read.load(
            "/mnt/spark-recommendation-models/neiss_model/neiss_tfidf.parquet"
        ).cache()
        model = BucketedRandomProjectionLSHModel.load(
            "/mnt/spark-recommendation-models/neiss_model/neiss_distance.model")
        vocab_model = CountVectorizerModel.load(
            "/mnt/spark-recommendation-models/neiss_model/neiss_vocab.model"
        )
        # define tokenizer pipeline
        tokenizer = Tokenizer(inputCol="narrative", outputCol="words")
        remover_words = StopWordsRemover(
            inputCol=tokenizer.getOutputCol(), outputCol="filtered")
        idf = IDF(inputCol="tf", outputCol="features")
        normalizer = Normalizer(inputCol=idf.getOutputCol(), outputCol="norm")
        pipeline = Pipeline(
            stages=[tokenizer, remover_words, vocab_model, idf, normalizer])

        # Get individual vector
        print(key)
        neiss_vector.show()
        tfidf_data = neiss_vector.where(
            neiss_vector.id == key).select('norm').collect()

        if len(tfidf_data) > 0:
            spark_key = neiss_vector.where(neiss_vector.id == key).select(
                'norm').collect()[0]['norm']
        else:
            body_dict = {"query": {
                "term": {'_id': key}
            }}
            es_result = es.search(index='ppe_cpsc_neiss', body=body_dict)
            data = create_narrative(es_result['hits']['hits'][0]['_source'])
            # Apply pipeline
            model_tfidf = pipeline.fit(data)
            data_tfidf = model_tfidf.transform(data)
            spark_key = data_tfidf.select('norm').collect()[0]['norm']

        # Find similarity

        nn_spark = model.approxNearestNeighbors(
            neiss_vector, spark_key, 21).select('id')
        nn_array = [str(row.id) for row in nn_spark.collect()]
        print(nn_array)

        return jsonify(
            status="success",
            key=key,
            data=nn_array[1:]
        )

    except IndexError as e:
        return jsonify(
            status="failed",
            key=key,
            data=e.args
        )

    except Exception as e:
        print("Unexpected error: check log file")
        print(e)
        return jsonify(
            status="failed",
            key=key,
            data="something went wrong"
        )


@spark_api.route('/ppe_eu_rapex/<key>/')
def spark_safetygate(key):
    '''
    Function : Find similarity Safety Gate
    Input    : key - _id ID
    Output   : JSON - _id Nearest Neighbors
    '''
    # Load computed spark dataframe and model

    try:
        # Load computed spark dataframe and model
        '''safetygate_vector = spark.read.load(
            "/mnt/spark-recommendation-models/safetygate_model/safetygate.tfidf.parquet").cache()
        model = BucketedRandomProjectionLSHModel.load(
            "/mnt/spark-recommendation-models/safetygate_model/safetygate.lsh.model.spark")'''
        tfidf_path = "/mnt/spark-recommendation-models/safetygate_model/safetygate.tfidf.parquet"
        model_path = "/mnt/spark-recommendation-models/safetygate_model/safetygate.lsh.model.spark"
        safetygate_vector, model = load_spark_data(tfidf_path, model_path)

        # Find 10 similarity
        spark_key = safetygate_vector.where(safetygate_vector.id == key).select(
            'norm').collect()[0]['norm']

        nn_spark = model.approxNearestNeighbors(
            safetygate_vector, spark_key, 21).select('id')
        nn_spark.show()

        nn_array = [str(row.id) for row in nn_spark.collect()]
        print(nn_array)
        print(safetygate_vector.count())

        return jsonify(
            status="success",
            key=key,
            data=nn_array[1:]
        )

    except IndexError as e:
        print("Index error", e)
        return jsonify(
            status="failed",
            key=key,
            data=e.args
        )

    except Exception as e:
        if "non-matching sizes" in str(e):
            safetygate_vector.unpersist()
            print("Cache CLEARED")
            spark_safetygate(key)
            return jsonify(
                status="cleared",
                key=key,
                data="Please refresh"
            )

        print("Unexpected error: check log file")
        print(e)
        return jsonify(
            status="failed",
            key=key,
            data="something went wrong"
        )


@spark_api.route('/ppe_phmsa/<key>/')
def spark_phmsa(key):
    '''
    Function : Find similarity NEISS
    Input    : key - _id ID 
    Output   : JSON - _id Nearest Neighbors 
    '''
    # Load computed spark dataframe and model

    try:
        # Load computed spark dataframe and model
        '''phmsa_vector = spark.read.load(
            "/mnt/spark-recommendation-models/phmsa_model/phmsa_tfidf.parquet").cache()
        model = BucketedRandomProjectionLSHModel.load(
            "/mnt/spark-recommendation-models/phmsa_model/phmsa_distance.model")'''

        tfidf_path = "/mnt/spark-recommendation-models/phmsa_model/phmsa_tfidf.parquet"
        model_path = "/mnt/spark-recommendation-models/phmsa_model/phmsa_distance.model"
        phmsa_vector, model = load_spark_data(tfidf_path, model_path)
        # Find 10 similarity
        spark_key = phmsa_vector.where(phmsa_vector.report_id == key).select(
            'norm').collect()[0]['norm']

        nn_spark = model.approxNearestNeighbors(
            phmsa_vector, spark_key, 21).select('report_id')
        nn_array = [str(row.report_id) for row in nn_spark.collect()]
        print(nn_array)

        return jsonify(
            status="success",
            key=key,
            data=nn_array[1:]
        )

    except IndexError as e:
        return jsonify(
            status="failed",
            key=key,
            data=e.args
        )

    except Exception as e:
        print("Unexpected error: check log file")
        print(e)
        return jsonify(
            status="failed",
            key=key,
            data="something went wrong"
        )


@spark_api.route('/ppe_cpsc_recall/<key>/')
def spark_recall(key):
    '''
    Function : Find similarity NEISS
    Input    : key - _id ID 
    Output   : JSON - _id Nearest Neighbors 
    '''
    # Load computed spark dataframe and model

    try:
        # Load computed spark dataframe and model
        '''recall_vector = spark.read.load(
            "/mnt/spark-recommendation-models/recalls_model/data.recalls.tfidf.parquet").cache()
        model = BucketedRandomProjectionLSHModel.load(
            "/mnt/spark-recommendation-models/recalls_model/model.recalls.similarity.spark")'''
        tfidf_path = "/mnt/spark-recommendation-models/recalls_model/data.recalls.tfidf.parquet"
        model_path = "/mnt/spark-recommendation-models/recalls_model/model.recalls.similarity.spark"
        recall_vector, model = load_spark_data(tfidf_path, model_path)

        # Find 10 similarity
        spark_key = recall_vector.where(recall_vector.RecallID == key).select(
            'norm').collect()[0]['norm']

        nn_spark = model.approxNearestNeighbors(
            recall_vector, spark_key, 21).select('RecallID')
        nn_array = [str(row.RecallID) for row in nn_spark.collect()]
        print(nn_array)

        return jsonify(
            status="success",
            key=key,
            data=nn_array[1:]
        )

    except IndexError as e:
        return jsonify(
            status="failed",
            key=key,
            data=e.args
        )

    except Exception as e:
        print("Unexpected error: check log file")
        print(e)
        return jsonify(
            status="failed",
            key=key,
            data="something went wrong"
        )


@spark_api.route('/ppe_cpsc_saferproducts/<key>/')
def spark_saferproducts(key):
    '''
    Function : Find similarity Safety Gate
    Input    : key - _id ID 
    Output   : JSON - _id Nearest Neighbors 
    '''
    # Load computed spark dataframe and model

    try:
        # Load computed spark dataframe and model
        print("Running SaferProducts Spark")
        '''saferproducts_vector = spark.read.load(
            # "/mnt/spark-recommendation-models/saferproducts-model/saferproducts_tfidf.parquet"
            "saferproducts_tfidf.parquet"
        ).cache()
        model = BucketedRandomProjectionLSHModel.load(
            # "/mnt/spark-recommendation-models/saferproducts-model/saferproducts_distance.model"
            "saferproducts_distance.model"
        )'''
        tfidf_path = "/mnt/spark-recommendation-models/saferproducts-model/saferproducts_tfidf.parquet"

        model_path = "/mnt/spark-recommendation-models/saferproducts-model/saferproducts_distance.model"

        saferproducts_vector, model = load_spark_data(tfidf_path, model_path)
        # Find 10 similarity
        spark_key = saferproducts_vector.where(saferproducts_vector.id == key).select(
            'norm').collect()[0]['norm']

        nn_spark = model.approxNearestNeighbors(
            saferproducts_vector, spark_key, 21).select('id')
        nn_array = [str(row.id) for row in nn_spark.collect()]
        print(nn_array)

        return jsonify(
            status="success",
            key=key,
            data=nn_array[1:]
        )

    except IndexError as e:
        return jsonify(
            status="failed",
            key=key,
            data=e.args
        )

    except Exception as e:
        if "non-matching sizes" in str(e):
            saferproducts_vector.unpersist()
            print("Cache CLEARED")
            spark_saferproducts(key)
            return jsonify(
                status="cleared",
                key=key,
                data="Please refresh"
            )

        print("Unexpected error: check log file")
        print(e)
        return jsonify(
            status="failed",
            key=key,
            data="something went wrong"
        )


@spark_api.route('/ppe_cpsc_ipii/<key>/')
def spark_ipii(key):
    '''
    Function : Find similarity Safety Gate
    Input    : key - _id ID 
    Output   : JSON - _id Nearest Neighbors 
    '''
    # Load computed spark dataframe and model

    try:
        # Load computed spark dataframe and model
        '''ipii_vector = spark.read.load(
            "/mnt/spark-recommendation-models/ipii_model/ipii_tfidf.parquet").cache()
        model = BucketedRandomProjectionLSHModel.load(
            "/mnt/spark-recommendation-models/ipii_model/ipii_distance.model")'''

        tfidf_path = "/mnt/spark-recommendation-models/ipii_model/ipii_tfidf.parquet"
        model_path = "/mnt/spark-recommendation-models/ipii_model/ipii_distance.model"
        ipii_vector, model = load_spark_data(tfidf_path, model_path)
        # Find 10 similarity
        spark_key = ipii_vector.where(ipii_vector.id == key).select(
            'norm').collect()[0]['norm']

        nn_spark = model.approxNearestNeighbors(
            ipii_vector, spark_key, 21).select('id')
        nn_array = [str(row.id) for row in nn_spark.collect()]
        print(nn_array)

        return jsonify(
            status="success",
            key=key,
            data=nn_array[1:]
        )

    except IndexError as e:
        return jsonify(
            status="failed",
            key=key,
            data=e.args
        )

    except Exception as e:
        print("Unexpected error: check log file")
        print(e)
        return jsonify(
            status="failed",
            key=key,
            data="something went wrong"
        )


@spark_api.route('/ppe_cpsc_dth/<key>/')
def spark_dth(key):
    '''
    Function : Find similarity Safety Gate
    Input    : key - _id ID 
    Output   : JSON - _id Nearest Neighbors 
    '''
    # Load computed spark dataframe and model

    try:
        # Load computed spark dataframe and model
        '''dth_vector = spark.read.load(
            "/mnt/spark-recommendation-models/dth_model/dth_tfidf.parquet").cache()
        model = BucketedRandomProjectionLSHModel.load(
            "/mnt/spark-recommendation-models/dth_model/dth_distance.model")'''
        tfidf_path = "/mnt/spark-recommendation-models/dth_model/dth_tfidf.parquet"
        model_path = "/mnt/spark-recommendation-models/dth_model/dth_distance.model"
        dth_vector, model = load_spark_data(tfidf_path, model_path)
        # Find 10 similarity
        spark_key = dth_vector.where(dth_vector.id == key).select(
            'norm').collect()[0]['norm']

        nn_spark = model.approxNearestNeighbors(
            dth_vector, spark_key, 21).select('id')
        nn_array = [str(row.id) for row in nn_spark.collect()]
        print(nn_array)

        return jsonify(
            status="success",
            key=key,
            data=nn_array[1:]
        )

    except IndexError as e:
        return jsonify(
            status="failed",
            key=key,
            data=e.args
        )

    except Exception as e:
        print("Unexpected error: check log file")
        print(e)
        return jsonify(
            status="failed",
            key=key,
            data="something went wrong"
        )


@spark_api.route('/ppe_health_canada/<key>/')
def spark_health_canada(key):
    '''
    Function : Find similarity Safety Gate
    Input    : key - _id ID
    Output   : JSON - _id Nearest Neighbors
    '''
    # Load computed spark dataframe and model

    try:
        # Load computed spark dataframe and model
        '''safetygate_vector = spark.read.load(
            "/mnt/spark-recommendation-models/safetygate_model/safetygate.tfidf.parquet").cache()
        model = BucketedRandomProjectionLSHModel.load(
            "/mnt/spark-recommendation-models/safetygate_model/safetygate.lsh.model.spark")'''
        tfidf_path = "/mnt/spark-recommendation-models/health_canada_model/health_canada_tfidf.parquet"
        model_path = "/mnt/spark-recommendation-models/health_canada_model/health_canada_distance.model"
        healt_canada_vector, model = load_spark_data(tfidf_path, model_path)
        # Find 10 similarity
        spark_key = healt_canada_vector.where(healt_canada_vector.id == key).select(
            'norm').collect()[0]['norm']

        nn_spark = model.approxNearestNeighbors(
            healt_canada_vector, spark_key, 21).select('id')
        nn_spark.show()

        nn_array = [str(row.id) for row in nn_spark.collect()]
        print(nn_array)

        return jsonify(
            status="success",
            key=key,
            data=nn_array[1:]
        )

    except IndexError as e:
        if "non-matching sizes" in str(e):
            healt_canada_vector.unpersist()
            print("Cache CLEARED")
            spark_safetygate(key)
            return jsonify(
                status="cleared",
                key=key,
                data="Please refresh"
            )
        return jsonify(
            status="failed",
            key=key,
            data=e.args
        )

    except Exception as e:
        print("Unexpected error: check log file")
        print(e)
        return jsonify(
            status="failed",
            key=key,
            data="something went wrong"
        )


@spark_api.route('/ppe_oecd_recalls/<key>/')
def spark_oecd(key):
    '''
    Function : Find similarity Safety Gate
    Input    : key - _id ID
    Output   : JSON - _id Nearest Neighbors
    '''
    # Load computed spark dataframe and model

    try:
        # Load computed spark dataframe and model
        '''safetygate_vector = spark.read.load(
            "/mnt/spark-recommendation-models/safetygate_model/safetygate.tfidf.parquet").cache()
        model = BucketedRandomProjectionLSHModel.load(
            "/mnt/spark-recommendation-models/safetygate_model/safetygate.lsh.model.spark")'''
        tfidf_path = "/mnt/spark-recommendation-models/oecd_model/oecd_recalls_tfidf.parquet"
        model_path = "/mnt/spark-recommendation-models/oecd_model/oecd_recalls_distance.model"
        oecd_vector, model = load_spark_data(tfidf_path, model_path)
        # Find 10 similarity
        spark_key = oecd_vector.where(oecd_vector.id == key).select(
            'norm').collect()[0]['norm']

        nn_spark = model.approxNearestNeighbors(
            oecd_vector, spark_key, 21).select('id')
        nn_spark.show()

        nn_array = [str(row.id) for row in nn_spark.collect()]
        print(nn_array)

        return jsonify(
            status="success",
            key=key,
            data=nn_array[1:]
        )

    except IndexError as e:
        if "non-matching sizes" in str(e):
            oecd_vector.unpersist()
            print("Cache CLEARED")
            spark_safetygate(key)
            return jsonify(
                status="cleared",
                key=key,
                data="Please refresh"
            )
        return jsonify(
            status="failed",
            key=key,
            data=e.args
        )

    except Exception as e:
        print("Unexpected error: check log file")
        print(e)
        return jsonify(
            status="failed",
            key=key,
            data="something went wrong"
        )
