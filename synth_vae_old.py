import os, time, argparse
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import BatchNormalization, Activation, Layer
from tensorflow.keras import metrics, optimizers
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.callbacks import Callback
import tensorflow.compat.v1.keras.backend as K
import glob as glob
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from statistics import mean

application_start_time = time.time()

print('Starting sample generation and validation')

def rand_frst(trn, val):
    val_acrcy_lst = []
    val_running_average = []

    X_trn = trn.iloc[:, 1:]
    y_trn = trn.iloc[:, 0]
    X_val = val.iloc[:, 1:]
    y_val = val.iloc[:, 0]

    for r in list(range(0, 10)):
        clf = RandomForestClassifier()
        clf.fit(X_trn, y_trn)
        val_raw_acc = accuracy_score(y_val, clf.predict(X_val))
        val_acrcy_lst.append(val_raw_acc)
        val_running_average.append(mean(val_acrcy_lst))
    return (val_running_average)


def synth_latent_3(latent_object, synth_index_name):
    print('Start synth sample gen from latent')
    synth_in_count = 3
    synth_sub_len = 200

    synth_ndx_strt = 0
    synth_full_frame = pd.DataFrame(columns=latent_object.columns)

    for subtype in sorted(latent_object.Labels.unique()):
        print(subtype)
        sub = latent_object[latent_object.Labels == subtype]
        print(synth_sub_len)
        synth_index = ['SYNTH-' + synth_index_name + '-' + jtem for jtem in [str(
            item).zfill(5) for item in list(range(synth_ndx_strt,
                                                  synth_sub_len + synth_ndx_strt))]]
        synth_sub_frame = pd.DataFrame(index=synth_index)
        synth_sub_frame.insert(0, 'Labels', sub.Labels[0])

        synth_dict = {}
        for synth_sample in synth_sub_frame.index:
            input_sample_set = sub.sample(synth_in_count)
            new_samp_vec = []
            for col in input_sample_set.iloc[:, 1:]:
                vals_inpt = input_sample_set.loc[:, col]
                choosen_val = vals_inpt.sample(1)
                new_samp_vec.append(choosen_val.values[0])

            synth_dict[synth_sample] = new_samp_vec
        synth_sub_frame = pd.concat([synth_sub_frame, pd.DataFrame(synth_dict).T], axis=1)

        synth_full_frame = pd.concat(
            [synth_full_frame, synth_sub_frame], axis=0)

        synth_ndx_strt = synth_ndx_strt + synth_sub_len
    print('Synthetic from latent done, ' + str(synth_sub_len) + ' samples generated for each subtype')
    return synth_full_frame


# Tybalt VAE; Way G., Greene, C., 2017


tf.compat.v1.disable_eager_execution()


def compute_latent(x):
    mu, sigma = x
    batch = K.shape(mu)[0]
    dim = K.shape(mu)[1]
    eps = K.random_normal(shape=(batch, dim), mean=0., stddev=1.0)
    return mu + K.exp(sigma / 2) * eps


class CustomVariationalLayer(Layer):
    """
    Define a custom layer
    """

    def __init__(self, **kwargs):
        self.is_placeholder = True
        super(CustomVariationalLayer, self).__init__(**kwargs)

    def vae_loss(self, x_input, x_decoded):
        reconstruction_loss = original_dim * metrics.binary_crossentropy(x_input, x_decoded)
        kl_loss = - 0.5 * K.sum(1 + z_log_var_encoded - K.square(z_mean_encoded) -
                                K.exp(z_log_var_encoded), axis=-1)
        return K.mean(reconstruction_loss + (K.get_value(beta) * kl_loss))

    def call(self, inputs):
        x = inputs[0]
        x_decoded = inputs[1]
        loss = self.vae_loss(x, x_decoded)
        self.add_loss(loss, inputs=inputs)
        return x


class WarmUpCallback(Callback):
    def __init__(self, beta, kappa):
        self.beta = beta
        self.kappa = kappa

    def on_epoch_end(self, epoch, logs={}):
        if K.get_value(self.beta) <= 1:
            K.set_value(self.beta, K.get_value(self.beta) + self.kappa)


parser = argparse.ArgumentParser()

parser.add_argument('version')
parser.add_argument('feature_file_path')
parser.add_argument('cohort_n_index', type=int)
parser.add_argument('trn_sz', type=int)
parser.add_argument('pre_train_epochs', type=int)
parser.add_argument('fine_tune_epochs', type=int)

args = parser.parse_args()

v = args.version
feature_files = args.feature_file_path
cohort_n_ndx = args.cohort_n_index
trn_size = args.trn_sz
pre_train_epochs = args.pre_train_epochs

inpt_val = pd.DataFrame()
dec_val = pd.DataFrame()
synth_lat_val = pd.DataFrame()
blend_val = pd.DataFrame()

file_paths = sorted(
    glob.glob(feature_files))
print('Total cohorts n = ', len(file_paths))

fine_tune_file = pd.read_csv(
    file_paths[cohort_n_ndx],
    sep='\t', index_col=0)
print(fine_tune_file.index.name)

out_dirs = ['/decoded_objs/',
            '/latent_objs/',
            '/loss_plots/',
            '/take-off_points/',
            '/synthetic_sample_sets/',
            '/rfe_out/']

# Build output dirs with <v> from command line
for out_dir in out_dirs:
    auto_path_name = 'i_o/' + v + fine_tune_file.index.name + out_dir
    print(auto_path_name)
    os.makedirs(os.path.dirname(auto_path_name), exist_ok=True)

pre_train_file = pd.DataFrame()

file_paths.remove(
    file_paths[cohort_n_ndx])

# Mean absolute deviation for feature selection on intersection of genes
# Normalized within each primary tumor type
print('Pre-train on cohorts n = ', len(file_paths))
for path in file_paths:
    file = pd.read_csv(path, sep='\t', index_col=0)
    pre_train_file = pd.concat([pre_train_file, file], axis=0)

# Each validation split constitutes an experimental replicate
vs_list = ['vs01@', 'vs02@', 'vs03@', 'vs04@', 'vs05@', 'vs06@', 'vs07@', 'vs08@', 'vs09@', 'vs10@',
           'vs11@', 'vs12@', 'vs13@', 'vs14@', 'vs15@', 'vs16@', 'vs17@', 'vs18@', 'vs19@', 'vs20@',
           'vs21@', 'vs22@', 'vs23@', 'vs24@', 'vs25@']

pre_train_loss_dict = {}
fine_tune_loss_dict = {}
for validation_split in vs_list:
    print(validation_split)
    val_split = validation_split + str(trn_size)
    trn = fine_tune_file.sample(trn_size)
    while_loop_val = 0
    while trn.Labels.value_counts().min() < 3:
        trn = fine_tune_file.sample(trn_size)
        while_loop_val += 1
        if while_loop_val == 50:
            break
    print('While undersampled loops: ', while_loop_val)
    if trn.Labels.value_counts().min() < 3:
        continue
    val = fine_tune_file.loc[fine_tune_file[~fine_tune_file.index.isin(trn.index)].index, :]
    inpt_val.insert(0, val_split, rand_frst(trn, val))

    # Begin training
    train_file = pre_train_file
    fit_on = str(len(pre_train_file))
    pre_trn = 'NONE'
    feature_set = feature_files.split('/')[-2]

    fine_tune_epochs = 'NA'
    features = train_file.columns[1:]

    original_dim = len(features)
    feature_dim = len(features)
    latent_dim = 250
    batch_size = 50

    encoder_inputs = keras.Input(shape=(feature_dim,))
    z_mean_dense_linear = layers.Dense(
        latent_dim, kernel_initializer='glorot_uniform', name="encoder_1")(encoder_inputs)
    z_mean_dense_batchnorm = layers.BatchNormalization()(z_mean_dense_linear)
    z_mean_encoded = layers.Activation('relu')(z_mean_dense_batchnorm)

    z_log_var_dense_linear = layers.Dense(
        latent_dim, kernel_initializer='glorot_uniform', name="encoder_2")(encoder_inputs)
    z_log_var_dense_batchnorm = layers.BatchNormalization()(z_log_var_dense_linear)
    z_log_var_encoded = layers.Activation('relu')(z_log_var_dense_batchnorm)

    latent_space = layers.Lambda(
        compute_latent, output_shape=(
            latent_dim,), name="latent_space")([z_mean_encoded, z_log_var_encoded])

    decoder_to_reconstruct = layers.Dense(
        feature_dim, kernel_initializer='glorot_uniform', activation='sigmoid')
    decoder_outputs = decoder_to_reconstruct(latent_space)

    learning_rate = 0.0005

    kappa = 1
    beta = K.variable(0)

    adam = optimizers.Adam(learning_rate=learning_rate)
    vae_layer = CustomVariationalLayer()([encoder_inputs, decoder_outputs])
    vae = Model(encoder_inputs, vae_layer)
    vae.compile(optimizer=adam, loss=None, loss_weights=[beta])

    pre_train_epochs = pre_train_epochs

    fit_start = time.time()
    history = vae.fit(train_file.iloc[:, 1:],
                      epochs=pre_train_epochs,
                      batch_size=batch_size,
                      shuffle=True,
                      callbacks=[WarmUpCallback(beta, kappa)],
                      verbose=0)
    pre_train_loss_dict[validation_split] = history.history['loss']
    fit_end = time.time() - fit_start

    # Fine tune
    train_file = trn
    pre_trn = 'TCGA_n=' + fit_on
    fit_on = trn.index.name
    fine_tune_epochs = args.fine_tune_epochs

    batch_size = 10
    history = vae.fit(train_file.iloc[:, 1:],
                      epochs=fine_tune_epochs, batch_size=batch_size, shuffle=True,
                      callbacks=[WarmUpCallback(beta, kappa)], verbose=0)
    fine_tune_loss_dict[validation_split] = history.history['loss']

    encoder = Model(encoder_inputs, z_mean_encoded)
    decoder_input = keras.Input(shape=(latent_dim,))
    _x_decoded_mean = decoder_to_reconstruct(decoder_input)
    decoder = Model(decoder_input, _x_decoded_mean)

    y_df = train_file.Labels
    decoded = pd.DataFrame(decoder.predict(encoder.predict(train_file.iloc[:, 1:])),
                           index=train_file.index, columns=train_file.iloc[:, 1:].columns)

    latent_object = pd.DataFrame(encoder.predict(train_file.iloc[:, 1:]),
                                 index=train_file.index)
    latent_object.index.name = trn.index.name
    latent_object = pd.concat([pd.DataFrame(y_df), latent_object], axis=1)

    decoded_labeled = pd.concat([pd.DataFrame(y_df), decoded], axis=1)
    decoded_labeled.to_csv(
        'i_o/' + v + '/' + fine_tune_file.index.name + '/decoded_objs/fit.' +
        fit_on + '_epochs.' + str(fine_tune_epochs) +
        '_pre_trained_on.' + pre_trn + '_epochs.' + str(pre_train_epochs) +
        '_decoded_obj_latent_dim.' + str(latent_dim) +
        '_' + feature_set + '_' + val_split + '.tsv', sep='\t')

    dec_val.insert(0, val_split, rand_frst(decoded_labeled, val))
    print('loop one done')

    synth_full_frame = synth_latent_3(latent_object, fine_tune_file.index.name)
    synth_lat_dec = pd.concat([synth_full_frame.iloc[:, 0],
                               pd.DataFrame(decoder.predict(synth_full_frame.iloc[:, 1:]),
                                            index=synth_full_frame.index)], axis=1)
    synth_lat_dec.columns = trn.columns
    synth_lat_dec.to_csv(
        'i_o/' + v + fine_tune_file.index.name + '/synthetic_sample_sets/fit.' + fit_on +
        '_epochs.' + str(fine_tune_epochs) + '_pre_trained_on.' + pre_trn + '_epochs.' +
        str(pre_train_epochs) + '_synthetic_sample_set_latent_dim.' + str(latent_dim) +
        '_' + feature_set + '_' + val_split + '.tsv', sep='\t')

    synth_lat_val.insert(0, val_split, rand_frst(synth_lat_dec, val))

    blend = pd.concat([trn, synth_lat_dec], axis=0)
    blend_val.insert(0, val_split, rand_frst(blend, val))

    blend_val.to_csv('i_o/' + v + fine_tune_file.index.name + '/take-off_points/' + val_split +
                     '_blend_val.tsv', sep='\t')

    inpt_val.to_csv('i_o/' + v + fine_tune_file.index.name + '/take-off_points/' + val_split +
                    '_input_val.tsv', sep='\t')

    dec_val.to_csv('i_o/' + v + fine_tune_file.index.name + '/take-off_points/' + val_split +
                   '_decoded_val.tsv', sep='\t')

    synth_lat_val.to_csv('i_o/' + v + fine_tune_file.index.name + '/take-off_points/' + val_split +
                         '_synth_lat_val.tsv', sep='\t')

pre_train_loss_frame = pd.DataFrame(pre_train_loss_dict)
fine_tune_loss_frame = pd.DataFrame(fine_tune_loss_dict)

pre_train_loss_frame.to_csv(
    'i_o/' + v + '/' + fine_tune_file.index.name + '/loss_plots/fit.' +
    fit_on + '_epochs.' + str(fine_tune_epochs) +
    '_pre_trained_on.' + pre_trn + '_epochs.' + str(pre_train_epochs) +
    '_pre_train_loss_vals_latent_dim.' + str(latent_dim) +
    '_' + feature_set + '.tsv', sep='\t')

fine_tune_loss_frame.to_csv(
    'i_o/' + v + '/' + fine_tune_file.index.name + '/loss_plots/fit.' +
    fit_on + '_epochs.' + str(fine_tune_epochs) +
    '_pre_trained_on.' + pre_trn + '_epochs.' + str(pre_train_epochs) +
    '_fine_tune_loss_vals_latent_dim.' + str(latent_dim) +
    '_' + feature_set + '.tsv', sep='\t')

print('main done')