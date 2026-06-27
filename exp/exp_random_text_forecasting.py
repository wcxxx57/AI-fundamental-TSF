from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast


class Exp_Random_Text_Forecast(Exp_Long_Term_Forecast):
    """Random-text negative-control forecasting experiment.

    This keeps the MM-TSFlib text/numeric fusion model unchanged, but reads a
    random text column. The default alignment is leak-free and comparable to the
    ECNU generated-text path: the text is read at the forecast origin row and
    the prior branch repeats an origin-known prior instead of reading the future
    prediction window.
    """

    def __init__(self, args):
        if not getattr(args, "text_column", ""):
            args.text_column = "Random_Text"
        if getattr(args, "text_origin_offset", None) is None:
            args.text_origin_offset = -1
        if not getattr(args, "prior_mode", ""):
            args.prior_mode = "origin_repeat"
        super().__init__(args)
