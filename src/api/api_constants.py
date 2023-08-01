
api_options = {
    #'outputs': ["depth"], # list of outputs to send in response. examples ["depth", "normalmap", 'heatmap', "normal", 'background_removed'] etc
    #'conversions': "", #TODO implement. it's a good idea to give some options serverside for because often that's challenging in js/clientside 
    'save':"" #TODO implement. To save on local machine. Can be very helpful for debugging.
}

# TODO: These are intended to be temporary
api_defaults={
    "BOOST": False,
    "NET_SIZE_MATCH": True
}

#These are enforced after user inputs
api_forced={
    "GEN_SIMPLE_MESH": False,
    "GEN_INPAINTED_MESH": False
}

#model diction TODO find a way to remove without forcing people do know indexes of models
models_to_index = {
    'res101':0, 
    'dpt_beit_large_512 (midas 3.1)':1,
    'dpt_beit_large_384 (midas 3.1)':2, 
    'dpt_large_384 (midas 3.0)':3,
    'dpt_hybrid_384 (midas 3.0)':4,
    'midas_v21':5, 
    'midas_v21_small':6,
    'zoedepth_n (indoor)':7, 
    'zoedepth_k (outdoor)':8, 
    'zoedepth_nk':9
}