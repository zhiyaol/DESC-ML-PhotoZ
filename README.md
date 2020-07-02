# DESC-ML-PhotoZ

In **Training-final.ipynb** and **Analysis-final.ipynb**, we apply and validate a Machine Learning Redshift Estimation algorithm to produce photometric redshift estimations for the simulated extragalactic catalog with magnitude errors *(cosmoDC2_v1.1.4_small_photoz_magerr_10y)* in LSST Dark Energy Science Collaboration (LSST DESC). We study the accuracy of our method using the known redshifts (also referred to as specz) in the cosmoDC2 simulations. 

**change_path.py** and **point_metrics.py** are supporting documents that are imported when we run the above-mentioned notebooks. **point_metrics.py** is imported from https://github.com/LSSTDESC/PZDC1paper/tree/master/metric_scripts on 10th May 2020. We made minor edits for it to compile in Python 3 environment.

**Analysis_old.ipynb** contains some code that are half developed and not useful to our project but might be useful in other projects.
