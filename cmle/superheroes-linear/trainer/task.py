import argparse
import model
import json
import os
import time

import tensorflow as tf
from tensorflow.python.lib.io.file_io import FileIO as open_file
from tensorflow.python.util import compat
from tensorflow.python.platform import gfile
from tensorflow.python.platform import tf_logging as logging

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
            '--MODEL_DIR',
            help='PATH OF OUTPUT DIR',
            required=True
    )
    parser.add_argument(
            '--job-dir',
            help='this model ignores this field, but it is required by gcloud',
            default='./junk'
    )
    parser.add_argument(
            '--TRAIN_STEPS',
            type=int,
            required=True
            #default=100000
    )


    # for hyper-parameter tuning
    parser.add_argument(
            '--BATCH_SIZE',
            help='Number of examples to compute gradient on',
            type=int,
            required=True
            #default=100
    )
#    parser.add_argument(
#            '--EMBEDDING_COLUMNS_SIZE',
#            default=4
#    )
#    parser.add_argument(
#            '--HIDDEN_UNITS',
#            help='Architecture of DNN part of wide-and-deep network',
#            default='1024,512,256'
#            #default='64,64,64,16,4'
#    )
    parser.add_argument(
            '--LEARNING_RATE',
            type=float,
            required=True
            #default=0.0606
    )
    parser.add_argument(
            '--L1_NORM',
            type=float,
            required=True
            #default=0.0606
    )
    parser.add_argument(
            '--L2_NORM',
            type=float,
            required=True
            #default=0.0606
    )

    # parse args
    args = parser.parse_args()
    arguments = args.__dict__
    
    print(str(arguments))

    # unused args provided by service
    arguments.pop('job-dir', None)
    arguments.pop('job_dir', None)
    MODEL_DIR = arguments.pop('MODEL_DIR')

    # when hp-tuning, we need to use different output directories for different runs
    output_dir = os.path.join(
            MODEL_DIR,
            json.loads(os.environ.get('TF_CONFIG', '{}')).get('task', {}).get('trial', '')
    )
 

    #COLUMNS=arguments["COLUMNS"].split(","), 
    #LABEL_FIELD=arguments["LABEL_FIELD"], 
    #FIELD_TYPES=dict(zip(arguments["COLUMNS"].split(","),arguments["FIELD_TYPES"].split(","))), 
    #with open_file(arguments["FIELD_CATEGORIES"]) as f:
    #    FIELD_CATEGORIES=json.load(f)
        
    #dict(zip(arguments["COLUMNS"].split(","),arguments["FIELD_CATEGORIES"].split(","))),
    LEARNING_RATE=arguments["LEARNING_RATE"]
    L1_NORM=arguments["L1_NORM"]
    L2_NORM=arguments["L2_NORM"]
    
    print("reading "+os.path.join(MODEL_DIR,"data","dataset_fields.json"))
    
    with open_file(os.path.join(MODEL_DIR,"data","dataset_fields.json"), "r") as f:
        config=json.load(f)
        COLUMNS=config["fields"]["columns"]
        FIELD_TYPES=config["fields"]["types"]
        FIELD_CATEGORIES=config["fields"]["categories"]
        LABEL_FIELD=config["label"]["column"]
        #FIELD_TYPES[LABEL_FIELD]=config["label"]["type"]
        #FIELD_CATEGORIES[LABEL_FIELD]=config["label"]["categories"]
        
        
    
    FIELD_DEFAULTS=[]
    for c in COLUMNS:
        if(FIELD_TYPES[c]=="bool"):
            FIELD_DEFAULTS.append([0])
        elif(FIELD_TYPES[c]=="string"):
            FIELD_DEFAULTS.append(["NA"])
        else:  
            FIELD_DEFAULTS.append([0.0])

    # run
    tf.logging.set_verbosity(tf.logging.INFO)
    # create estimator
    feature_columns, estimator = model.build_model(COLUMNS, LABEL_FIELD, FIELD_TYPES, FIELD_CATEGORIES, output_dir, LEARNING_RATE, L1_NORM, L2_NORM)

    #estimator = tf.contrib.estimator.add_metrics(estimator, model.my_rmse)

    # read_dataset(MODEL_DIR, FIELD_DEFAULTS, COLUMNS, LABEL_FIELD, mode=tf.contrib.learn.ModeKeys.EVAL, BATCH_SIZE=100, TRAIN_STEPS=100000)
    train_spec = tf.estimator.TrainSpec(input_fn=model.read_dataset(
        MODEL_DIR, FIELD_DEFAULTS, COLUMNS, LABEL_FIELD,
        BATCH_SIZE=arguments['BATCH_SIZE'], 
        TRAIN_STEPS=arguments['TRAIN_STEPS'],
        mode=tf.estimator.ModeKeys.TRAIN),
        max_steps=arguments['TRAIN_STEPS']
    )

    eval_spec = tf.estimator.EvalSpec(input_fn=model.read_dataset(MODEL_DIR, FIELD_DEFAULTS, COLUMNS, LABEL_FIELD,
        BATCH_SIZE=arguments['BATCH_SIZE'], 
        TRAIN_STEPS=arguments['TRAIN_STEPS'],
        mode=tf.estimator.ModeKeys.EVAL),
        #steps = None,
        start_delay_secs = 2*60,#20 * 60, # start evaluating after N seconds
        throttle_secs = 1*60)#10 * 60)    # evaluate every N seconds

    print("launching train")
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
    print("train done")
    
    
    def _get_timestamped_export_dir(export_dir_base):
        # When we create a timestamped directory, there is a small chance that the
        # directory already exists because another worker is also writing exports.
        # In this case we just wait one second to get a new timestamp and try again.
        # If this fails several times in a row, then something is seriously wrong.
        max_directory_creation_attempts = 10

        attempts = 0
        while attempts < max_directory_creation_attempts:
            export_timestamp = int(time.time())

            export_dir = os.path.join(
                    compat.as_bytes(export_dir_base), compat.as_bytes(
                            str(export_timestamp)))
            if not gfile.Exists(export_dir):
                # Collisions are still possible (though extremely unlikely): this
                # directory is not actually created yet, but it will be almost
                # instantly on return from this function.
                return export_dir
            time.sleep(1)
            attempts += 1
            logging.warn(
                    "Export directory {} already exists; retrying (attempt {}/{})".format(
                            export_dir, attempts, max_directory_creation_attempts))
        raise RuntimeError("Failed to obtain a unique export directory name after "
                                             "{} attempts.".format(max_directory_creation_attempts))
    
    try:
        feature_spec = tf.feature_column.make_parse_example_spec(feature_columns)
        export_input_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(feature_spec)
        #export_input_fn = tf.contrib.learn.build_parsing_serving_input_fn(feature_spec)
        servable_model_path=estimator.export_savedmodel(os.path.join(output_dir,"model"),export_input_fn)
            #_get_timestamped_export_dir(os.path.join(output_dir,"model")),export_input_fn)
    except:
        pass
    
    
    #estimator.export_savedmodel(os.path.join(output_dir,'Servo'),serving_input_receiver_fn=model.serving_input_fn())