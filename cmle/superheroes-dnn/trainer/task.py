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
    )


    # for hyper-parameter tuning
    parser.add_argument(
            '--BATCH_SIZE',
            help='Number of examples to compute gradient on',
            type=int,
            required=True
    )
    parser.add_argument(
            '--EMBEDDING_COLUMNS_SIZE',
            required=True
            #default=4
    )
    parser.add_argument(
            '--HIDDEN_UNITS',
            help='Architecture of DNN part of wide-and-deep network',
            required=True
    )
    parser.add_argument(
            '--LEARNING_RATE',
            type=float,
            required=True
    )
    parser.add_argument(
            '--L1_NORM',
            type=float,
            required=True
    )
    parser.add_argument(
            '--L2_NORM',
            type=float,
            required=True
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

    LEARNING_RATE=arguments["LEARNING_RATE"]
    L1_NORM=arguments["L1_NORM"]
    L2_NORM=arguments["L2_NORM"]
    HIDDEN_UNITS=arguments["HIDDEN_UNITS"]
    EMBEDDING_COLUMNS_SIZE=arguments["EMBEDDING_COLUMNS_SIZE"]

    with open_file(os.path.join(MODEL_DIR,"data","dataset_fields.json"), "r") as f:
        config=json.load(f)
        COLUMNS=config["fields"]["columns"]
        FIELD_TYPES=config["fields"]["types"]
        FIELD_CATEGORIES=config["fields"]["categories"]
        LABEL_FIELD=config["label"]["column"]
        
        
    
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
    feature_columns, estimator = model.build_model(COLUMNS, LABEL_FIELD, FIELD_TYPES, FIELD_CATEGORIES, output_dir, LEARNING_RATE, L1_NORM, L2_NORM, EMBEDDING_COLUMNS_SIZE, HIDDEN_UNITS)

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
        start_delay_secs = 2*60,#20 * 60, # start evaluating after N seconds
        throttle_secs = 1*60)#10 * 60)    # evaluate every N seconds

    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

    
    try:
        feature_spec = tf.feature_column.make_parse_example_spec(feature_columns)
        export_input_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(feature_spec)
        servable_model_path=estimator.export_savedmodel(os.path.join(output_dir,"model"),export_input_fn)
    except:
        pass
    