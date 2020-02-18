import copy
import torch
import torch.nn as nn
import warnings

def transform_layer(input, from_inst, to_inst, args={}, attrs={}):
    if isinstance(input, from_inst) :
        for key in args.keys() :
            arg = args[key]
            if isinstance(arg, str) :
                if arg.startswith(".") :
                    args[key] = getattr(input, arg[1:])
                    
        output = to_inst(**args)
        
        for key in attrs.keys() :
            attr = attrs[key]
            if isinstance(attr, str) :
                if attr.startswith(".") :
                    attrs[key] = getattr(input, attr[1:])
        
            setattr(output, key, attrs[key])
    else :
        output = input        
    return output

def transform_model(input, from_inst, to_inst, args={}, attrs={}, inplace=True, _warn=True):
    if inplace :
        output = input
        if _warn :
            warnings.warn("\n * Caution : The Input Model is CHANGED because inplace=True.", Warning)
    else :
        output = copy.deepcopy(input)
        
    if isinstance(output, from_inst) :
        output = transform_layer(output, from_inst, to_inst, args, attrs)
    for name, module in output.named_children() :
        setattr(output, name, transform_model(module, from_inst, to_inst, args, attrs, _warn=False))
        
    # Depreciated
#     for name, m in output.named_children() :
#         if isinstance(m, nn.Sequential) :
#             for i, layer in enumerate(m) :
#                 m[i] = transform_layer(layer, from_inst, to_inst, args, attrs)
#         else :
#             m = transform_layer(m, from_inst, to_inst, args, attrs)
    return output