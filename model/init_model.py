from model.FPTrans_AdaptiveFSS.FPTrans import FPTrans_AdaptiveFSS
from model.DCAMA_AdaptiveFSS.model import DCAMA_AdaptiveFSS
from model.MSANet_AdaptiveFSS.MSANet_Adapter import OneModel as MSANet_AdaptiveFSS

__networks = {
    'fptrans_adaptivefss': FPTrans_AdaptiveFSS,
    'dcama_adaptivefss': DCAMA_AdaptiveFSS,
    'msanet_adaptivefss': MSANet_AdaptiveFSS,
}

def load_model(con):
    if con.model_name.lower() in __networks:
        model = __networks[con.model_name.lower()](con)
        return model
    else:
      raise ValueError(f'Not supported network: {con.network}. {list(__networks.keys())}')