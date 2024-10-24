from feature_mask_module import FM
from bayesian_feature_mask_module import BFM
from mcdropout_feature_mask_module import MCFM


def build_selection_layer(method=None, reg=None, num_target_fea=None, **kwargs):
    """
    Build a variable selection layer.
    :param method: The name of the variable selection method.
    :param reg: The regularization term for the variable selection method, default=None.
    :param num_target_fea: The number of the features to be selected.
    :param kwargs: Only works for FM (the number of hidden neurons for FM => for insight study of FM).
    :return: A variable seleciton layer.
    """
    if method in ['FM']:
        x = FM(name=method, lat_dim=kwargs['lat_dim'])  # lat_dim
    elif method in ['BFM']:
        x = BFM(name=method, input_dim=kwargs['input_dim'],
                lat_dim=kwargs['lat_dim'], kl_weight=kwargs['kl_weight'])
    elif method in ['MCFM']:
        x = MCFM(name=method, lat_dim=kwargs['lat_dim'])
    else:
        raise ValueError('Please enter a valid feature selector name...')
    return x
