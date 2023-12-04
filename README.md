# Dynamic Flat-Field Correction

This package implements the dynamic flat-field correction.

## Brief description

The method described here consists of two separate steps:

1. Initially, reference flat-fields and dark-fields are
acquired and PCA is used to obtain the most relevant principal
components of the flat-field dataset.

2. During data acquisition with a sample, the effecitve flat-
field is computed for each individual frame as a weighted sum
of principal components, while the weights subject to minimize
the total variance of the corrected image.

## How to cite

S. Birnsteinova *et. al.* Online dynamic flat-field correction
for MHz microscopy data at European XFEL (2023). J. Synchrotron
Rad. 30, 1030-1037. DOI: [10.1107/S1600577523007336](
http://dx.doi.org/10.1107/S1600577523007336)
