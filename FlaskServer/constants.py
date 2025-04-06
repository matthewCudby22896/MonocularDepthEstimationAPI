import logging

# MARIGOLD (LCM)
DEFAULT_DENOISING_STEPS  = 4
DEFAULT_ENSEMBLE_SIZE = 1

METRIC_3D = 'metric3d'
MARIGOLD = 'marigold'
MIDAS = 'midas'

MODEL_OPTIONS = {
    METRIC_3D: {
        'small': 'metric3d_vit_small',
        'large': 'metric3d_vit_large',
        'giant': 'metric3d_vit_giant2',
    },
    MARIGOLD: {
        'default': 'default'
    }, 
    MIDAS: {
        'default' : 'default'
    }
}

logging.basicConfig(
    level=logging.DEBUG,
    format='[%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler()]
)

logger = logging.getLogger(__name__)