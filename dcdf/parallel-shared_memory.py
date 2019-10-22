from typing import Any, Optional, Callable, List, Dict, Tuple
import multiprocessing as mp
from multiprocessing import shared_memory
import itertools as it
import numpy as np
import pandas as pd
import nibabel as nib

from dcdf.data import get_datapoints, get_subject_cdf2

# man I hate having to use global variables ....
# open to alternative suggestions ...
_binsize=None
_lowerlimit=None
_func_dict=None
_filter=None
_shm_ref_name=None
_shm_ref_shape=None
_shm_ref_dtype=None
_shm_mask_name=None
_shm_mask_shape=None
_shm_mask_dtype=None

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
        mask_indices = np.where(mask != 0)[0]
        shared_mask_mem = shared_memory.SharedMemory(create=True,size=mask_indices.nbytes)
        shared_mask = np.ndarray(
            mask_indices.shape,
            dtype=mask_indices.dtype,
            buffer=shared_mask_mem.buf
        ) 
        shared_mask[:] = mask_indices[:]
        shm_mask_tuple = (shared_mask_mem.name,mask_indices.shape,mask_indices.dtype)
    else:
        shm_mask_tuple = (None,None,None)

    # Now we need to setup the shared reference ...
    # The cumcount array will be passed as shared memory
    # The rest as as explicit arguments ...
    shared_ref_mem = shared_memory.SharedMemory(create=True,size=reference.cumcount.nbytes)
    shared_ref = np.ndarray(
        reference.cumcount.shape,
        dtype=reference.cumcount.dtype,
        buffer=shared_ref_mem.buf
    )
    shared_ref[:] = reference.cumcount[:]
    shm_ref_tuple = (shared_ref_mem.name,reference.cumcount.shape,reference.cumcount.dtype)

    # Prep the arguments tuples
    args = zip(
        subjects_list,
        it.repeat(indv_mask_list) if indv_mask_list is None else indv_mask_list,
    )
    pool = mp.Pool(n_procs,_worker_init,(reference.binsize,reference.lowerlimit,func_dict,shm_ref_tuple,shm_mask_tuple,filter))
    results = pool.starmap(_mp_measure,args)

    # Don't forget to free our shared resources
    del shared_mask
    del shared_ref
    shared_mask_mem.close()
    shared_ref_mem.close()
    shared_mask_mem.unlink()
    shared_ref_mem.unlink()

    return pd.DataFrame(data=results,index=subjects_list,columns=sorted(func_dict.keys()))

def _worker_init(binsize: np.float32,
        lowerlimit: np.float32,
        func_dict: Dict[str,Callable[[np.ndarray,np.ndarray,np.float32],np.float32]],
        shm_ref_tuple: Tuple[str,Tuple[int],np.dtyper],
        shm_mask_tuple: Tuple[str,Tuple[int],np.dtype],
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
    global _shm_ref_name
    global _shm_ref_shape
    global _shm_ref_dtype
    global _shm_mask_name
    global _shm_mask_shape
    global _shm_mask_dtype
    _filter = filter
    _binsize = binsize
    _lowerlimit = lowerlimit
    _func_dict = dict(func_dict)
    _shm_ref_name,_shm_ref_shape,_shm_ref_dtype=shm_ref_tuple
    _shm_mask_name,_shm_mask_shape,_shm_mask_dtype=shm_mask_tuple


def _mp_measure(subject: str,
             indv_mask_filename: Optional[str]=None,
    ) -> List[np.float32]:
    """
    Function to apply provided measures to a single subject

    :param subjects: Nifti file paths 
    :param indv_mask_filename: Mask to be applied to subject image
    """
    # Get our datapoints, using the group-level mask if specified, otherwise individual masks
    if _shm_mask_name is not None:
        shm_mdx = shared_memory.SharedMemory(name=_shm_mask_name)
        midx = np.ndarray(_shm_mask_shape,dtype=_shm_mask_dtype,buffer=shm_mdx.buf) 
        subject_data = get_datapoints(subject,mask_indices=midx,filter=_filter)
    else:
        subject_data = get_datapoints(subject,
                mask_filename=indv_mask_filename,
                filter=_filter
        )

    # Get our shared reference data
    shm_ref = shared_memory.SharedMemory(name=_shm_ref_name)
    ref = np.ndarray(_shm_ref_shape,dtype=_shm_ref_dtype,buffer=shm_ref.buf)

    # Get the subject_cdf and compute the difference from the reference CDF
    subject_cdf = get_subject_cdf2(subject_data,len(ref),_lowerlimit,_binsize)

    sub = subject_cdf.cumcount
    # Calculate each of the requested results and append to the dataframe
    return [_func_dict[f](sub,ref,_binsize) for f in sorted(_func_dict.keys())]
