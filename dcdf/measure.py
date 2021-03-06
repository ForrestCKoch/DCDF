from typing import Any, Optional, Callable, List, Dict, Tuple
import os
from dcdf.data import get_subject_cdf, get_datapoints
import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.stats.stats import CumfreqResult
import nibabel as nib

def get_func_dict(func_file: str) -> dict:
    """
    Reads a textfile which should be in the format:
    [function name] : [function code]
    where [function code] will later be executed using `eval`

    :param func_file: name of the textfile containing the equations

    :returns: a dictionary of [function name] and [function code] pairs
    """

    d = {}
    with open(func_file,'r') as fh:
        for line in fh:
            x = line.rstrip('\n').split(':')
            d[x[0]]=x[1]   
    return d

def measure_single_subject(subject: str,
                   reference: CumfreqResult,
                   func_dict: Dict[str,Callable[[np.ndarray,np.ndarray,np.float32],np.float32]],
                   indv_mask: Optional[str]=None,
                   group_mask_indices: Optional[np.ndarray]=None,
                   filter: Optional[Callable[[np.ndarray],np.ndarray]]=None,
                   _print_inverse: Optional[bool]=False
    ) -> Tuple[str,Dict[str,np.float32]]:
    """
    Function to apply provided measures to a single subject

    :param subjects: Nifti file paths 
    :param reference: CumfreqResult from `data.get_reference_cdf`
    :param func_dict: Output of `measure.get_func_dict`.  A dictionary
        of functions to be calculated over CDF differences.  Keys will be used 
        as column names in the return of this function
    :param indv_mask: Mask to be applied to subject image
    :param group_mask_filename: If not None, this should be a path to a nifti 
        file which will be used as a mask for eac of the individual images.  
        If set, `indv_mask_list` will be ignored.
    :param filter: Optional: function which takes in an np.ndarray and
        returns an np.ndarray.  Can be used to apply a filter to the data 
        (e.g thresholding)

    :returns: a Tuple with first element being the nifti file path, and the
        second element being a dictionary of results with keys being function 
        names
    """
    # Get our datapoints, using the group-level mask if specified, otherwise individual masks
    if group_mask_indices is not None:
        subject_data = get_datapoints(subject,
                mask_indices=group_mask_indices,
                filter=filter
        )
    else:
        subject_data = get_datapoints(subject,
                mask_filename=indv_mask,
                filter=filter
        )

    # Get the subject_cdf and compute the difference from the reference CDF
    subject_cdf = get_subject_cdf(subject_data,reference)

    #cdf_diff = reference.cumcount - subject_cdf.cumcount

    sub = subject_cdf.cumcount
    ref = reference.cumcount
    bs = reference.binsize
    subi = subject_cdf.inverse
    refi = reference.inverse
    bsi = reference.inverse_binsize
    # Calculate each of the requested results and append to the dataframe
    #subj_results = {f: func_dict[f](sub,ref,bs) for f in func_dict.keys()}
    #subj_results = {f: (lambda s,r,b:eval(fun)
    # Decalre our subject's results dictionary

    if _print_inverse:
        print(','.join([subject]+[str(x) for x in subi]))
        return (subject,{})
    else:
        subj_results = {}
        for f in func_dict.keys(): 
            #print(f)
            # create our lambda function from the string
            # only allowing access to numpy, s, r, & b
            func = lambda s,r,b,si,ri,bi:eval(func_dict[f],{'np':np,'s':s,'r':r,'b':b,'si':si,'ri':ri,'bi':bi}) 
            subj_results[f] = func(sub,ref,bs,subi,refi,bsi)

        return (subject,subj_results)


def measure_subjects(subjects_list: List[str], 
                     reference: CumfreqResult, 
                     func_dict: Dict[str,Callable[[np.ndarray,np.ndarray,np.float32],np.float32]],
                     indv_mask_list: Optional[List[str]]=None,
                     group_mask_filename: Optional[str]=None,
                     filter: Optional[Callable[[np.ndarray],np.ndarray]]=None
    ) -> pd.DataFrame:
    """
    Wrapper around measure_single_subject to apply to each subject

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

    :returns: pandas.DataFrame containing all of the results.
    """

    # Prepare the dataframe we will return
    results = pd.DataFrame(data=None, columns=['nifti']+list(func_dict.keys())).set_index('nifti')

    # If we are using one mask for everybody, prepare it now
    if group_mask_filename is not None:
        mask = nib.load(group_mask_filename).get_fdata().flatten()
        mask_indices = np.where(mask != 0)

    for i in range(0,len(subjects_list)):
        subject = subjects_list[i]


        _,subj_results = measure_single_subject(subject=subject,
                reference=reference,
                func_dict=func_dict,
                indv_mask=indv_mask_list[i] if indv_mask_list is not None else None,
                group_mask_indices=mask_indices if group_mask_filename is not None else None,
                filter=filter
        )
        results = results.append(pd.Series(subj_results, name=subject))
    
    return results

def print_measurements(mdf: pd.DataFrame):
    """
    This function will print out the results of `measure.measure_subjects`.

    :param mdf: pd.DataFrame returned from `measure.measure_subjects`
    """
    print(','.join(['nifti']+list(mdf.keys())))
    for i in range(0,len(mdf)):
        print(','.join([mdf.index[i]]+[str(mdf[k][i]) for k in mdf.keys()]))

