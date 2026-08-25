import yaml
from FeatureExtractrionFunctions import *



def load_config(config_file):
    with open(config_file, 'r') as stream:
        try:
            config = yaml.safe_load(stream)
            return config
        except yaml.YAMLError as exc:
            print(exc)



config = load_config("config_feat_extraction.yaml")
df_config = config['data_and_features']

sel_states   = df_config.get('sel_states', 'tubeleak')
sel_feature = df_config['selected_feature']

# pull only the parameter block for the selected feature
feat_params = (df_config.get('feature_params') or {}).get(sel_feature, {})

kwargs_arguments = {
    'Fs':             df_config['Fs'],
    'downsample':     df_config['downsample'],
    'summarize':      df_config.get('summarize', None),
    'feature_params': feat_params,
    'base_folder':    df_config.get('base_folder', ''),
    'save_folder':    df_config.get('save_folder', 'saved_features'),
}

perform_feature_extraction(
    sel_states   = sel_states,
    sel_feature = sel_feature,
    noises      = df_config.get('noises', ['lab']),
    takes       = df_config.get('takes', ['1', '2']),
    microphones = df_config.get('microphones', ['1l', '1m']),
    **kwargs_arguments,
)






