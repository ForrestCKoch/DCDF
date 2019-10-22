from typing import Any, Optional, Callable, List

import pickle

import nibabel as nib
import numpy as np
import scipy.stats as stats
from scipy.stats.stats import CumfreqResult

def get_datapoints(
        input_filename:str,
        mask_filename: Optional[str]=None,
        mask_indices: Optional[np.ndarray]=None,
        filter: Optional[Callable[[np.ndarray],np.ndarray]]=None
    )->np.ndarray:
    """
    This function reads a nifti file and returns a flat (1D) array of
    the image.  Various options can be used to filter the array

    :param input_filename: filename of the nifti file to be loaded
    :param mask_filename: Optional: filename of mask to be applied to data
    :param mask_indices: Optional & ignored if mask_filename is set.
    Indices to extract from data array
    :param filter: Optional: function which takes in an np.ndarray and
    returns an np.ndarrray.  Can be used to apply a filter to the data
    (e.g thresholding)
    """
    img = nib.load(input_filename)
    img_data = img.get_fdata().flatten()
    if mask_filename is not None:
        mask_img = nib.load(mask_filename)
        mask_data = mask_img.get_fdata().flatten()
        img_data = img_data[mask_data!=0]
    elif mask_indices is not None: 
        img_data = img_data[mask_indices]

    if filter is not None:
        img_data = filter(img_data)

    return img_data

def get_reference_cdf(
        reference_list: List[str],
        numbins: Optional[int]=1000,
        indv_mask_list: Optional[List[str]]=None,
        group_mask_filename: Optional[str]=None,
        filter: Optional[Callable[[np.ndarray],np.ndarray]]=None
    )->CumfreqResult:
    """
    This function will return a CDF to be used as a reference based on the
    provided images.
    :param reference_list: List of nifti files to be used for reference.
    :param numbins: How many bins should be used for the reference
    :param indv_mask_list: A list with the same length as `reference_list` of masks
    to be used for each subject.
    :param group_mask_filename: If not None, this should be a path to a nifti
    file which should be used as a mask for each of the reference images. 
    If set, `indv_mask_list` will be ignored.
    :param filter: A filtering function to be applied to the flattened array of nifti data.
    """

    if indv_mask_list is not None and len(indv_mask_list) != len(reference_list):
        raise ValueError('indv_mask_list and reference_list have different lengths')


    if group_mask_filename is not None:
        group_mask = nib.load(group_mask_filename).get_fdata().flatten()
        mask_indicies = np.where(group_mask != 0)
        pooled_points = np.concatenate([get_datapoints(r,mask_indicies=mask_indicies) for r in reference_list])
    elif indv_mask_list is not None:
        img_pairs = zip(reference_list, indv_mask_list)
        pooled_points = np.concatenate([get_datapoints(r,mask_filename=m) for r,m in img_pairs])
    else:
        pooled_points = np.concatenate([get_datapoints(r) for r in reference_list])

    if filter is not None:
        pooled_points = filter(pooled_points)

    return stats.cumfreq(pooled_points,
                        numbins=numbins,
                        weights=np.repeat(1/len(pooled_points),len(pooled_points))
                        )

def save_reference(reference: CumfreqResult ,filename: str):
    """
    Save the reference using pickle. If available, protocol 4 will be used.
    :param reference: CumfreqResult to be saved.
    :param filename: path to save the reference to.
    """
    with open(filename,'wb') as fh:
        pickle.dump(reference,fh,protocol=min(4,pickle.HIGHEST_PROTOCOL))

def load_reference(filename) -> CumfreqResult:
    """
    Load and retrun a pickled reference
    :param filename: path to the pickled CumfreqResult which should be loaded.
    """
    with open(filename,'rb') as fh:
        return pickle.load(fh)
    return None

def get_subject_cdf(subject_array: np.ndarray, 
        reference_cdf: CumfreqResult) -> CumfreqResult:
    """
    Calculate the individual subject's cdf with respect to the reference CDF.
    :param subject_array: numpy array of datapoints from `get_datapoints`
    :param reference_cdf: reference_cdf that was built using `get_reference_cdf`.
    """
    numbins = len(reference_cdf.cumcount)
    lowerlimit = reference_cdf.lowerlimit
    binsize = reference_cdf.binsize
    upperlimit = lowerlimit + (binsize * numbins)

    subject_cdf = stats.cumfreq(subject_array,
                                numbins=numbins,
                                defaultreallimits=(lowerlimit,upperlimit),
                                weights=np.repeat(1/len(subject_array),len(subject_array))
                                )
    return subject_cdf

def get_subject_cdf2(subject_array: np.ndarray, 
            numbins: int, 
            lowerlimit: np.float32, 
            binsize: int
    ) -> CumfreqResult:
    """
    Calculate the individual subject's cdf with respect to the reference CDF.
    :param subject_array: numpy array of datapoints from `get_datapoints`
    :param numbins: len(Cumfreqresult.cumcount)
    :param lowerlimit: CumfreqResult.lowerlimit
    :param binsize: Cumfreqresult.binsize
    """
    upperlimit = lowerlimit + (binsize * numbins)

    subject_cdf = stats.cumfreq(subject_array,
                                numbins=numbins,
                                defaultreallimits=(lowerlimit,upperlimit),
                                weights=np.repeat(1/len(subject_array),len(subject_array))
                                )
    return subject_cdf
