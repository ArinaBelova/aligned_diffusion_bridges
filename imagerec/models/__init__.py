from imagerec.models.unet import ConditionalUNet
from sbalign.utils.sb_utils import get_timestep_embedding
from scripts.imagerec.options import dict_to_namespace

def build_model_from_args(args):
    # a timestep embedding is implemented in UNet already
    # timestep_embed_fn = get_timestep_embedding(embedding_type=args.timestep_embed_type,
    #                                            embedding_dim=args.timestep_embed_dim)
    args = dict_to_namespace(args)
    if args.which_model_G == "ConditionalUNet":
        # need to extract input_nc etc. correctly
        model = ConditionalUNet(
            in_nc=args.setting.in_nc, 
            out_nc=args.setting.out_nc,
            nf=args.setting.nf,
            depth=args.setting.depth, # default=4, unet depth
        )

    else:
        raise ValueError(f"Model of type {args.which_model_G} is not supported.")
                
    return model