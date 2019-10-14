from typing import Any, Optional, Callable, List, Dict
import os
from data import get_subject_cdf, get_datapoints
import numpy as np
import pandas as pd
import scipy.stats as stats
import nibabel as nib

def get_func_dict():
    return {'linearDCDF':lambda x:np.sum(x)}

def measure_subjects(subjects_list: List[str], 
                     reference: stats.CumfreqResult, 
                     func_dict: Dict[str,Callable[[np.ndarray],np.float32]],
                     indv_mask_list: Optional[List[str]]=None,
                     group_mask_filename: Optional[str]=None,
                     filter: Optional[Callable[[np.ndarray],np.ndarray]]=None
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
    """

    # Prepare the dataframe we will return
    results = pd.Dataframe(data=None, columns=['nifti']+list(func_dict.keys())).set_index('nifti')

    # If we are using one mask for everybody, prepare it now
    if group_mask_filename is not None:
        mask = nib.load(group_mask_filename).get_fdata().flatten()
        mask_indices = np.where(mask != 0)

    for i in range(0,len(subjects_list)):
        subject = subjects_list[i]

        # Get our datapoints, using the group-level mask if specified, otherwise individual masks
        if group_mask_filename is not None:
            subject_data = get_datapoints(subject,filter=filter)[mask_indices]
        else:
            mask_file = indv_mask_list[i] if indv_mask_list is not None else None
            subject_data = get_datapoints(subject,mask_file,filter=filter)

        # Get the subject_cdf and compute the difference from the reference CDF
        subject_cdf = get_subject_cdf(subject_data,reference)
        cdf_diff = reference.cumcount - subject.cumcount

        # Calculate each of the requested results and append to the dataframe
        subj_results = {f: func_dict[f](cdf_diff) for f in func_dict.keys()}
        results.append(pd.Series(subj_results, name=subject))
    
    return results

def print_measurements(mdf: pd.DataFrame):
    """
    This function will print out the results of `measure.measure_subjects`.
    :param mdf: pd.DataFrame returned from `measure.measure_subjects`
    """
    print(','.join(list(mdf.keys())))
    for i in range(0,len(mdf)):
        print(','.join([mdf[k][i] for k in mdf.keys()]))

