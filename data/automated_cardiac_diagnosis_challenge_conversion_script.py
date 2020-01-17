#!/usr/bin/env python
# -*- coding: utf-8 -*-

from glob import glob
import re
from os import path
import numpy as np
from pandas import DataFrame
import nibabel as nib


n2i = {'rv_endo': 1,
       'lv_epi': 2,
       'lv_endo': 3}

df = DataFrame()
id_finder = re.compile('.*\/patient(?P<patient_id>\d{3})\/.*')

for fn in glob('training/*/*.cfg'):
    # get unique id for case
    root, _ = path.split(fn)
    vals = {}
    r = id_finder.match(fn)
    vals['id'] = r.group('patient_id')

    # read meta information file for case
    meta = {}
    fh = open(fn, 'r')
    for line in fh.read().splitlines():
        label, val = line.split(':', 1)
        meta[label.strip()] = val.strip()
    
    vals['group']  = meta['Group']
    vals['height'] = float(meta['Height'])
    vals['weight'] = float(meta['Weight'])

    # load end-systolic and end-diastolic segmentation masks
    idx_ed = '%02i' % int(meta['ED'])
    idx_es = '%02i' % int(meta['ES'])
    mask_ed = nib.load(path.join(root, f'patient{vals["id"]}_frame{idx_ed}_gt.nii.gz'))
    mask_es = nib.load(path.join(root, f'patient{vals["id"]}_frame{idx_es}_gt.nii.gz'))
    ma_ed = mask_ed.get_data()
    ma_es = mask_es.get_data()

    # conversion factor for number of voxels in ml (mm^3 -> ml)
    conversion_factor = np.prod(mask_ed.header.get_zooms()[:3]) / 1000

    # end-systolic volume (ml)
    vals['lv_esv'] = np.sum(ma_es == n2i['lv_endo']) * conversion_factor
    vals['rv_esv'] = np.sum(ma_es == n2i['rv_endo']) * conversion_factor

    # end-diastolic volume (ml)
    vals['lv_edv'] = np.sum(ma_ed == n2i['lv_endo']) * conversion_factor
    vals['rv_edv'] = np.sum(ma_ed == n2i['rv_endo']) * conversion_factor

    # stroke volume (ml)
    vals['lv_sv'] = vals['lv_edv'] - vals['lv_esv']
    vals['rv_sv'] = vals['rv_edv'] - vals['rv_esv']

    # ejection fraction (%)
    vals['lv_ef'] = vals['lv_sv'] / vals['lv_edv'] * 100
    vals['rv_ef'] = vals['rv_sv'] / vals['rv_edv'] * 100

    # ventricular mass (mg)
    vals['lv_vm'] = np.sum(ma_es == n2i['lv_epi']) * conversion_factor * 1.05

    df = df.append(vals, ignore_index=True)

df = df.round(decimals=1)

print(df)
df.to_csv('automated_cardiac_diagnosis_challenge.csv', index=False)