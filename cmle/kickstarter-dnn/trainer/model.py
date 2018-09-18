from tensorflow.contrib.learn.python.learn.utils import saved_model_export_utils
import tensorflow.contrib.metrics as tfmetrics
import tensorflow as tf
import numpy as np
import os

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



def get_features(COLUMNS, LABEL_FIELD, FIELD_TYPES, FIELD_CATEGORIES, EMBEDDING_COLUMNS_SIZE):
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
    
