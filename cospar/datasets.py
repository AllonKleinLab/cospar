

from pathlib import Path, PurePath
import scanpy as sc
from . import settings
from . import logging as logg

url_prefix='https://kleintools.hms.harvard.edu/tools/downloads/cospar'

def synthetic_bifurcation(data_des='bifur'):
    """
    Synthetic clonal dataset with static barcoding.

    We simulated a differentiation process over a bifurcation fork. 
    Cells are barcoded in the beginning, and the barcodes remain unchanged.  
    In the simulation, we resample clones over time, 
    like the experimental design to obtain the hematopoietic dataset 
    or the reprogramming dataset. The dataset has two time points.  

    Parameters
    ----------
    data_des: `str`
        A key to label this dataset. 
    """

    data_path=settings.data_path
    figure_path=settings.figure_path
    #data_name='bifurcation_static_BC_adata_preprocessed.h5ad'
    data_name='bifur_adata_preprocessed.h5ad'
    return load_data_core(data_path,figure_path,data_name,data_des)

# def synthetic_bifurcation_dynamic_BC(data_des='bifur_conBC'):
#     """
#     Synthetic clonal dataset with dynamic barcoding.

#     We simulated a differentiation process over a bifurcation fork. 
#     Cells are barcoded, and the barcodes could accumulate mutations, which we call 
#     `dynamic barcoding`. In the simulation, we resample clones over time, 
#     like the experimental design to obtain the hematopoietic dataset 
#     or the reprogramming dataset. The dataset has two time points. 

#     Parameters
#     ----------
#     data_des: `str`
#         A key to label this dataset. 
#     """

#     data_path=settings.data_path
#     figure_path=settings.figure_path
#     data_name='bifurcation_dynamic_BC_adata_preprocessed.h5ad'
#     return load_data_core(data_path,figure_path,data_name,data_des)


def reprogramming(data_des='CellTagging'):
    """
    The reprogramming dataset from 

    * Biddy, B. A. et al. `Single-cell mapping of lineage and identity in direct reprogramming`. Nature 564, 219–224 (2018).

    This dataset has multiple time points for both the clones and the state measurements. 
    The cells are barcoded over 3 rounds during the entire differentiation process. 
    There are multiple ways to assemble the barcodes on day 0, day 3, and day 13 
    into a clonal ID. Below, we provide three variants:

    * Concatenate barcodes on day 0 and day 13, as in the original 
      analysis (adata.obsm['X_clone_Concat_D0D3'], the default);

    * Concatenate barcodes on day 0, day 3, and day 13 (adata.obsm['X_clone_Concat_D0D3D13']);

    * No concatenation; each cell has up to 3 barcodes (adata.obsm['X_clone_NonConcat_D0D3D13']).

    The last choice keeps the nested clonal structure in the data. 
    You can choose any one of the clonal arrangement for downstream analysis, 
    by setting `adata_orig.obsm['X_clone']=adata_orig.obsm['X_clone_Concat_D0D3']`. 
    The three clonal arrangements give very similar fate prediction.

    Parameters
    ----------
    data_des: `str`
        A key to label this dataset. 
    """

    data_path=settings.data_path
    figure_path=settings.figure_path
    data_name='CellTagging_adata_preprocessed.h5ad'
    return load_data_core(data_path,figure_path,data_name,data_des)


def reprogramming_Day0_3_28(data_des='Reprog_128'):
    """
    The reprogramming dataset from 

    * Biddy, B. A. et al. `Single-cell mapping of lineage and identity in direct reprogramming`. Nature 564, 219–224 (2018).

    This dataset has time points on day 0, day 3, and day 28. Only day 28 has clonal information. 
    The cells are barcoded over 3 rounds during the entire differentiation process. 
    There are multiple ways to assemble the barcodes on day 0, day 3, and day 13 
    into a clonal ID. Below, we concatenate barcodes on day 0 and day 13, 
    as in the original analysis.


    Parameters
    ----------
    data_des: `str`
        A key to label this dataset. 
    """

    data_path=settings.data_path
    figure_path=settings.figure_path
    data_name='Reprog_128_D0D3_adata_preprocessed.h5ad'
    return load_data_core(data_path,figure_path,data_name,data_des)


# def reprogramming_static_BC(data_des='CellTagging'):
#     """
#     The reprogramming dataset from 

#     * Biddy, B. A. et al. `Single-cell mapping of lineage and identity in direct reprogramming`. Nature 564, 219–224 (2018).

#     This dataset has multiple time points for both the clones and the state measurements. 

#     The cells are barcoded over 3 rounds during the entire differentiation process. 
#     We combine up to 3 tags from the same cell into a single clonal label in representing 
#     the X_clone matrix. In this representation, each cell has at most one clonal label. 
#     Effectively, we convert the barcodes into static labels that do not carry temporal information.
    
#     Parameters
#     ----------
#     data_des: `str`
#         A key to label this dataset. 
#     """

#     data_path=settings.data_path
#     figure_path=settings.figure_path
#     data_name='CellTagging_ConcatenateClone_adata_preprocessed.h5ad'
#     return load_data_core(data_path,figure_path,data_name,data_des)

# def reprogramming_dynamic_BC(data_des='CellTagging_NoConcat'):
#     """
#     The reprogramming dataset from 

#     * Biddy, B. A. et al. `Single-cell mapping of lineage and identity in direct reprogramming`. Nature 564, 219–224 (2018).

#     This dataset has multiple time points for both the clones and the state measurements. 

#     The cells are barcoded over 3 rounds during the entire differentiation process. 
#     We treat barcode tags from each round as independent clonal label here. In this 
#     representation, each cell can have multiple clonal labels at different time points.
    
#     Parameters
#     ----------
#     data_des: `str`
#         A key to label this dataset. 
#     """

#     data_path=settings.data_path
#     figure_path=settings.figure_path
#     data_name='CellTagging_NoConcat_adata_preprocessed.h5ad'
#     return load_data_core(data_path,figure_path,data_name,data_des)

def lung(data_des='Lung'):
    """
    The direct lung differentiation dataset from 

    * Hurley, K. et al. Cell Stem Cell (2020) doi:10.1016/j.stem.2019.12.009.

    This dataset has multiple time points for the state manifold, but only one time point
    for the clonal observation on day 27. 
    
    Parameters
    ----------
    data_des: `str`
        A key to label this dataset. 
    """

    data_path=settings.data_path
    figure_path=settings.figure_path
    data_name='Lung_pos17_21_D27_adata_preprocessed.h5ad'
    return load_data_core(data_path,figure_path,data_name,data_des)

def hematopoiesis(data_des='LARRY'):
    """
    The hematopoiesis data set from 

    * Weinreb, C., Rodriguez-Fraticelli, A., Camargo, F. D. & Klein, A. M. Science 367, (2020)

    This dataset has 3 time points for both the clonal and state measurements. It only contains cells with clonal 
    labels, which has ~50000 cells. Running the whole pipeline for the first time could take several hours in 
    a standard personal computer. 
    
    Parameters
    ----------
    data_des: `str`
        A key to label this dataset. 
    """

    data_path=settings.data_path
    figure_path=settings.figure_path
    data_name='LARRY_adata_preprocessed.h5ad'
    return load_data_core(data_path,figure_path,data_name,data_des)

def hematopoiesis_130K(data_des='LARRY'):
    """
    The hematopoiesis data set from 

    * Weinreb, C., Rodriguez-Fraticelli, A., Camargo, F. D. & Klein, A. M. Science 367, (2020)

    This dataset has 3 time points for both the clonal and state measurements. It includes
    cells with or without clonal labels, which has a total of ~130K cells. 
    
    Parameters
    ----------
    data_des: `str`
        A key to label this dataset. 
    """

    data_path=settings.data_path
    figure_path=settings.figure_path
    data_name='Complete_LARRY_dataset_adata_preprocessed.h5ad'
    return load_data_core(data_path,figure_path,data_name,data_des)

def hematopoiesis_Gata1_states(data_des='LARRY_Gata1_lineage'):
    """
    All of the hematopoiesis data set from 

    * Weinreb, C., Rodriguez-Fraticelli, A., Camargo, F. D. & Klein, A. M. Science 367, (2020)

    This dataset includes non-clonally-labeled states that express Gata1. In total, it has ~38K cells.
    
    Parameters
    ----------
    data_des: `str`
        A key to label this dataset. 
    """

    data_path=settings.data_path
    figure_path=settings.figure_path
    data_name='LARRY_Gata1_lineage_adata_preprocessed.h5ad'
    return load_data_core(data_path,figure_path,data_name,data_des)


def hematopoiesis_subsampled(data_des='LARRY_sp500_ranking1'):
    """
    Top 15% most heterogeneous clones of the hematopoiesis data set from 

    * Weinreb, C., Rodriguez-Fraticelli, A., Camargo, F. D. & Klein, A. M. Science 367, (2020)

    This dataset has 3 time points for both the clones and the state measurements. 
    This sub-sampled data better illustrates the power of CoSpar in robstly 
    inferring differentiation dynamics from a noisy clonal dataset. Also, it 
    is smaller and thus much faster to analyze. 
    
    Parameters
    ----------
    data_des: `str`
        A key to label this dataset. 
    """

    data_path=settings.data_path
    figure_path=settings.figure_path
    data_name='LARRY_sp500_ranking1_adata_preprocessed.h5ad'
    return load_data_core(data_path,figure_path,data_name,data_des)

def load_data_core(data_path,figure_path,data_name,data_des):
    url=f'{url_prefix}/{data_name}'
    path=f'{data_path}/{data_name}'
    path=Path(path)
    figure_path=Path(figure_path)

    if not path.parent.is_dir():
        logg.info(f'creating directory {path.parent}/ for saving data')
        path.parent.mkdir(parents=True)

    if not figure_path.is_dir():
        logg.info(f'creating directory {figure_path}/ for saving figures')
        figure_path.mkdir(parents=True)

    #print(url)
    status=_check_datafile_present_and_download(path,backup_url=url)
    if status:
        adata=sc.read(path)
        #adata.uns['data_path']=[str(data_path)]
        #adata.uns['figure_path']=[str(figure_path)]
        adata.uns['data_des']=[str(data_des)]
        return adata
    else:
        logg.error("Error, files do not exist")
        return None    

def _check_datafile_present_and_download(path, backup_url=None):
    """Check whether the file is present, otherwise download."""
    path = Path(path)
    if path.is_file():
        return True
    if backup_url is None:
        return False
    logg.info(
        f'try downloading from url\n{backup_url}\n'
        '... this may take a while but only happens once'
    )
    if not path.parent.is_dir():
        logg.info(f'creating directory {path.parent}/ for saving data')
        path.parent.mkdir(parents=True)

    _download(backup_url, path)
    return True


def _download(url: str, path: Path):
    try:
        import ipywidgets
        from tqdm.auto import tqdm
    except ImportError:
        from tqdm import tqdm

    from urllib.request import urlopen, Request

    blocksize = 1024 * 8
    blocknum = 0

    try:
        with urlopen(Request(url, headers={"User-agent": "scanpy-user"})) as resp:
            total = resp.info().get("content-length", None)
            with tqdm(
                unit="B",
                unit_scale=True,
                miniters=1,
                unit_divisor=1024,
                total=total if total is None else int(total),
            ) as t, path.open("wb") as f:
                block = resp.read(blocksize)
                while block:
                    f.write(block)
                    blocknum += 1
                    t.update(len(block))
                    block = resp.read(blocksize)

    except (KeyboardInterrupt, Exception):
        # Make sure file doesn’t exist half-downloaded
        if path.is_file():
            path.unlink()
        raise