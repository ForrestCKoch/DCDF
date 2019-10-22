from typing import Any, Optional, Callable, List, Dict, Tuple
import multiprocessing as mp
import itertools as it
import numpy as np
import pandas as pd
import nibabel as nib

from dcdf.data import get_datapoints, get_subject_cdf2

from scipy.stats.stats import CumfreqResult

# man I hate having to use global variables ....
# open to alternative suggestions ...
_binsize=None
_lowerlimit=None
_func_dict=None
_filter=None
_shared_ref=None
_shared_mask=None

def parallel_measure_subjects(subjects_list: List[str],
                   reference: CumfreqResult,
                   func_dict: Dict[str,Callable[[np.ndarray,np.ndarray,np.float32],np.float32]],
                   indv_mask_list: Optional[List[str]]=None,
                   group_mask_filename: Optional[str]=None,
                   filter: Optional[Callable[[np.ndarray],np.ndarray]]=None,
                   n_procs: Optional[int]=None,
    ) -> pd.DataFrame:
    """
    :param subjects_list: List of nifti file paths 
    :param reference: CumfreqResult from `data.get_reference_cdf`
    :param func_dict: Output of `measure.get_func_dict`.  A dictionary
    of functions to be calculated over CDF differences.  Keys will be used as column names
    in the return of this function
    :param indv_mask_list: A list with the same length as `subjects_list` to be used for each subject.
    :param group_mask_filename: If not None, this should be a path to anifti file which will be
    used as a mask for eac of the individual images.  If set, `indv_mask_list` will be ignored.
    :param filter: Optional: function which takes in an np.ndarray and
    returns an np.ndarray.  Can be used to apply a filter to the data 
    (e.g thresholding)
    :param n_procs: Number of processes to be started.  If none, then the number returned by `os.cpu_count()` is used
    """
    # Prepare the dataframe we will return
    #results = pd.DataFrame(data=None, columns=['nifti']+list(func_dict.keys())).set_index('nifti')

    # If we are using one mask for everybody, prepare it now
    if group_mask_filename is not None:
        mask = nib.load(group_mask_filename).get_fdata().flatten()
        shared_mask = np.where(mask != 0)[0]
    else:
        shared_mask = None

    # Prep the arguments tuples
    args = zip(
        subjects_list,
        it.repeat(indv_mask_list) if indv_mask_list is None else indv_mask_list,
    )
    pool = mp.Pool(n_procs,_worker_init,(reference.binsize,reference.lowerlimit,func_dict,shared_mask,reference.cumcount,filter))
    results = pool.starmap(_mp_measure,args)

    return pd.DataFrame(data=results,index=subjects_list,columns=sorted(func_dict.keys()))

def _worker_init(binsize: np.float32,
        lowerlimit: np.float32,
        func_dict: Dict[str,Callable[[np.ndarray,np.ndarray,np.float32],np.float32]],
        shared_mask: Tuple[str,Tuple[int],np.dtype],
        shared_ref: Tuple[str,Tuple[int],np.dtype],
        filter: Optional[Callable[[np.ndarray],np.ndarray]]=None
    ) -> None:
    """
    Initialization function for parallel workers calling `_mp_measure`.
    :param binsize: `CumfreqResult.binsize`
    :param lowerlimit: `CumfreqResult.lowerlimit`.
    :param func_dict: Output of `measure.get_func_dict`.  A dictionary
    :param shm_ref_tuple: (shm.name,shape,dtype)
    :param shm_mask_tuple: (shm.name,shape,dtype)
    :param filter: Optional: function which takes in an np.ndarray and
    """
    global _binsize
    global _lowerlimit
    global _func_dict
    global _filter
    global _shared_ref
    global _shared_mask
    _filter = filter
    _binsize = binsize
    _lowerlimit = lowerlimit
    _func_dict = dict(func_dict)
    _shared_ref = np.copy(shared_ref)
    _shared_mask = np.copy(shared_mask) if shared_mask is not None else None
    #_shared_ref = shared_ref
    #_shared_mask = shared_mask


def _mp_measure(subject: str,
             indv_mask_filename: Optional[str]=None,
    ) -> List[np.float32]:
    """
    Function to apply provided measures to a single subject

    :param subjects: Nifti file paths 
    :param indv_mask_filename: Mask to be applied to subject image
    """
    # Get our datapoints, using the group-level mask if specified, otherwise individual masks
    if _shared_mask is not None:
        subject_data = get_datapoints(subject,mask_indices=_shared_mask,filter=_filter)
    else:
        subject_data = get_datapoints(subject,
                mask_filename=indv_mask_filename,
                filter=_filter
        )

    # Get our shared reference data
    ref = _shared_ref

    # Get the subject_cdf and compute the difference from the reference CDF
    subject_cdf = get_subject_cdf2(subject_data,len(ref),_lowerlimit,_binsize)

    sub = subject_cdf.cumcount
    # Calculate each of the requested results and append to the dataframe
    return [_func_dict[f](sub,ref,_binsize) for f in sorted(_func_dict.keys())]
