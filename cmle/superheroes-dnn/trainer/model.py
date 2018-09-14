from tensorflow.contrib.learn.python.learn.utils import saved_model_export_utils
import tensorflow.contrib.metrics as tfmetrics
import tensorflow as tf
import numpy as np
import os

#CSV_COLUMNS    = ('ontime,dep_delay,taxiout,distance,avg_dep_delay,avg_arr_delay' + \
#                                ',carrier,dep_lat,dep_lon,arr_lat,arr_lon,origin,dest').split(',')
#LABEL_COLUMN = 'ontime'
#DEFAULTS         = [[0.0],[0.0],[0.0],[0.0],[0.0],[0.0],\
#                                ['na'],[0.0],[0.0],[0.0],[0.0],['na'],['na']]

def read_dataset(MODEL_DIR, FIELD_DEFAULTS, COLUMNS, LABEL_FIELD, BATCH_SIZE, TRAIN_STEPS, mode=tf.contrib.learn.ModeKeys.EVAL):
     
    
    def create_trainset():
        ds = tf.data.TextLineDataset(os.path.join(MODEL_DIR,"data","tf_trainset.csv")).skip(1)
        def _parse_line(line):
            # Decode the line into its fields
            fields = tf.decode_csv(line, FIELD_DEFAULTS)

            # Pack the result into a dictionary
            features = dict(zip(COLUMNS,fields))

            # Separate the label from the features
            label = features.pop(LABEL_FIELD)

            return features, label

        parsed_ds = ds.map(_parse_line)

        return parsed_ds.shuffle(TRAIN_STEPS).repeat().batch(BATCH_SIZE)

    #with open_file(os.path.join(MODEL_DIR,"data","tf_evalset.csv"), "w") as f:
    #    df_eval[COLUMNS].to_csv(f, index=False)

    def create_evalset():
        ds = tf.data.TextLineDataset(os.path.join(MODEL_DIR,"data","tf_evalset.csv")).skip(1)
        def _parse_line(line):
            # Decode the line into its fields
            fields = tf.decode_csv(line, FIELD_DEFAULTS)

            # Pack the result into a dictionary
            features = dict(zip(COLUMNS,fields))

            # Separate the label from the features
            label = features.pop(LABEL_FIELD)

            return features, label

        parsed_ds = ds.map(_parse_line)

        return parsed_ds.shuffle(TRAIN_STEPS).repeat().batch(BATCH_SIZE)
    
    return create_trainset if mode == tf.contrib.learn.ModeKeys.TRAIN else create_evalset



def get_features_raw(origin_file, dest_file): # NOT USED FOR NOW
    real = {
        colname : tf.feature_column.numeric_column(colname) \
                for colname in \
                    ('dep_delay,taxiout,distance,avg_dep_delay,avg_arr_delay' + 
                     ',dep_lat,dep_lon,arr_lat,arr_lon').split(',')
    }
    sparse = {
        'carrier': tf.feature_column.categorical_column_with_vocabulary_list('carrier',
                                vocabulary_list='AS,B6,WN,HA,OO,F9,NK,EV,DL,UA,US,AA,MQ,VX'.split(','),
                                dtype=tf.string)
        , 'origin': tf.feature_column.categorical_column_with_vocabulary_file('origin',origin_file)
        , 'dest'     : tf.feature_column.categorical_column_with_vocabulary_file('dest',dest_file)
    }
    return real, sparse

def get_features(COLUMNS, LABEL_FIELD, FIELD_TYPES, FIELD_CATEGORIES, EMBEDDING_COLUMNS_SIZE):
    #return get_features_raw(origin_file, dest_file)
    feature_columns=[]
    for c in COLUMNS:
        if c == LABEL_FIELD:
            continue

        if FIELD_TYPES[c]=="string":
            cat=tf.feature_column.categorical_column_with_vocabulary_list(
                key=c,
                vocabulary_list=list(FIELD_CATEGORIES[c])
            )
            emb=tf.feature_column.embedding_column(cat,float(EMBEDDING_COLUMNS_SIZE))
            """
                SELF NOTE:
                with this signature: tf.feature_column.embedding_column(cat,EMBEDDING_COLUMNS_SIZE)
                EMBEDDING_COLUMNS_SIZE is an integer, and I get the following error:
                    "a float is required"
                the documentation (https://www.tensorflow.org/api_docs/python/tf/feature_column/embedding_column) states:
                tf.feature_column.embedding_column(
                    categorical_column,
                    dimension,
                    ...
                 )
                    Args:
                        categorical_column: A _CategoricalColumn created by a categorical_column_with_* function. This column produces the sparse IDs that are inputs to the embedding lookup.
                        dimension: An integer specifying dimension of the embedding, must be > 0.
                        
                regardless of this, when invoked with tf.feature_column.embedding_column(cat,float(EMBEDDING_COLUMNS_SIZE)), it does not throw any errors..
            """
            feature_columns.append(emb)
        if FIELD_TYPES[c]=="number":
            feature_columns.append(tf.feature_column.numeric_column(key=c))
    
    return feature_columns
    

def parse_hidden_units(s):
    return [int(item) for item in s.split(',')]

def build_model(COLUMNS, LABEL_FIELD, FIELD_TYPES, FIELD_CATEGORIES, MODEL_DIR, LEARNING_RATE, L1_NORM, L2_NORM, EMBEDDING_COLUMNS_SIZE, HIDDEN_UNITS):
    
    feature_columns = get_features(COLUMNS, LABEL_FIELD, FIELD_TYPES, FIELD_CATEGORIES, EMBEDDING_COLUMNS_SIZE)
    
    HIDDEN_UNITS=parse_hidden_units(HIDDEN_UNITS)

    return feature_columns, tf.estimator.DNNClassifier(
        HIDDEN_UNITS,
        feature_columns,
        n_classes=len(FIELD_CATEGORIES[LABEL_FIELD]),
        label_vocabulary=list(FIELD_CATEGORIES[LABEL_FIELD]),
        model_dir=MODEL_DIR,
        optimizer=tf.train.FtrlOptimizer(learning_rate=LEARNING_RATE,
                                        l1_regularization_strength=L1_NORM, 
                                        l2_regularization_strength=L2_NORM)
    )
    

#def create_embed(sparse_col):
#    dim = 10 # default
#    if hasattr(sparse_col, 'bucket_size'):
#         nbins = sparse_col.bucket_size
#         if nbins is not None:
#                dim = 1 + int(round(np.log2(nbins)))
#    return tf.feature_column.embedding_column(sparse_col, dimension=dim)

def serving_input_fn():
    feature_placeholders = {
        key : tf.placeholder(tf.float32, [None]) \
            for key in ('dep_delay,taxiout,distance,avg_dep_delay,avg_arr_delay' +
                     ',dep_lat,dep_lon,arr_lat,arr_lon').split(',')
    }
    feature_placeholders.update( {
        key : tf.placeholder(tf.string, [None]) \
            for key in 'carrier,origin,dest'.split(',')
    } )

    features = {
        key: tf.expand_dims(tensor, -1)
        for key, tensor in feature_placeholders.items()
    }
    return tf.estimator.export.build_raw_serving_input_receiver_fn(feature_placeholders)

def my_rmse(predictions, labels, **args):
    prob_ontime = predictions['probabilities'][:,1]

    return {'rmse': tf.metrics.root_mean_squared_error(prob_ontime, labels)}