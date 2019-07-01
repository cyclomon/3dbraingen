# Official Pytorch Implementation of "Generation of 3D Brain MRI Using Auto-Encoding Generative Adversarial Networks" (accepted by MICCAI 2019)

This repository provides a PyTorch implementation of 3D brain Generation. It can successfully generates plausible 3-dimensional brain MRI with Generative Adversarial Networks. Trained models are also provided in this page.

## Paper
Generation of 3D Brain MRI Using Auto-Encoding Generative Adversarial Networks. The 22nd International Conference on Medical Image Computing and Computer Assisted Intervention(MICCAI 2019)

## Dependencies
* [Python 3.5+](https://www.continuum.io/downloads)
* [PyTorch 0.4.0+](http://pytorch.org/)
* [Jupyter Notebook](https://jupyter.org/)
* [Nilearn](https://nilearn.github.io/)
* [Nibabel](https://nipy.org/nibabel/)

We highly recommend you to use Jupyter Notebook for the better visualization!

## Dataset
You can download the Normal MRI data in [Alzheimer's Disease Neuroimaging Initiative(ADNI)](http://adni.loni.usc.edu/)
, Tumor MRI data in [BRATS2018](https://www.med.upenn.edu/sbia/brats2018/data.html) and Stroke MRI data in [Anatomical Tracings of Lesions After Stroke (ATLAS)](http://fcon_1000.projects.nitrc.org/indi/retro/atlas.html)

We converted all the DICOM(.dcm) files of ADNI into Nifti(.nii) file format using [SPM12](https://www.fil.ion.ucl.ac.uk/spm/software/spm12/) I/O tools.
