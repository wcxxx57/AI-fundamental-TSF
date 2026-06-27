from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast


class Exp_LLM_Generated_Forecast(Exp_Long_Term_Forecast):
    """LLM-generated self-text forecasting experiment.

    The model and training loop are inherited from the existing text/numeric
    fusion path. This wrapper only changes the defaults that matter for the
    leak-free ECNU-generated dataset:
    - read `ECNU_LLM_Text` unless the caller supplies another text column;
    - align text to the forecast origin row (`s_end - 1`);
    - use an origin-known prior instead of future-window prior values.
    """

    def __init__(self, args):
        if not getattr(args, "text_column", ""):
            args.text_column = "ECNU_LLM_Text"
        if getattr(args, "text_origin_offset", None) is None:
            args.text_origin_offset = -1
        if not getattr(args, "prior_mode", ""):
            args.prior_mode = "origin_repeat"
        super().__init__(args)
