import mir_eval
import numpy as np


def compute_validation(reference_sources, estimated_sources, reference_mixed_source):
    """
    :param reference_sources: ndarray shape = (nsrc, nsample)
    :param estimated_sources: ndarray shape = (nsrc, nsample)
    :param reference_mixed_source: ndarray shape= (1, nsample)
    :return: nsdr, sir, sar
    """
    nsrc = reference_sources.shape[0]
    mixed_source = np.tile(reference_mixed_source, (nsrc, 1))  # shape = (nsrc, nsample)
    try:
        mir_eval.separation.validate(reference_sources,estimated_sources)
        mir_eval.separation.validate(mixed_source, reference_sources)
        [sdr, sir, sar, _] = mir_eval.separation.bss_eval_sources(reference_sources, estimated_sources)
        [sdr_raw, _, _, _] = mir_eval.separation.bss_eval_sources(mixed_source, reference_sources)
        return [sdr / sdr_raw, sir, sar]
    except:
        return [None, None, None]
