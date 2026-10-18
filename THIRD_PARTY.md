# Third-party materials and provenance

The MIT license at this repository root covers our code. Our new paper,
notebook text and figures use CC BY 4.0 (LICENSE-CONTENT.md). Neither license
relicenses third-party inputs. Bundled copies retain their original terms.

## PAT16 imaging and supplied assessments

The copied BIDS dataset identifies its license as **CC0** in
`data/bids/dataset_description.json`. Source datasets: Aerts and Marinazzo,
BTC_preop, OpenNeuro ds001226 **v5.0.0**; BTC_postop, ds002080 **v4.0.0**.
The accompanying reference derivatives and assessment fields are preserved from
the public source release, with its notices in `data/source_context/`.

- https://doi.org/10.18112/openneuro.ds001226.v5.0.0
- https://doi.org/10.18112/openneuro.ds002080.v4.0.0
- https://doi.org/10.1038/s41597-022-01806-4
- https://github.com/arvidl/glioma-brain-digital-twin-public

The source public repository revision resolved during preparation was
`d0159ec273ac808d5f9b039a823e88e3b887d67b`. The actual copied bytes are fixed by
`data/manifest.json`; no assertion is made that a later checkout of the upstream
repository exactly reproduces this local release. No unreleased patient data or
additional participant records are added. These remain exploratory research
materials; no individual diagnosis or clinically validated prediction is supplied.

## Schaefer atlas and Yeo labels

Schaefer100/7-network atlas, CBIG tag
`v0.14.3-Update_Yeo2011_Schaefer2018_labelname`:
https://github.com/ThomasYeoLab/CBIG . Retain `data/atlas/CBIG_LICENSE.md`
and `data/atlas/NOTICE.md`. Cite Schaefer et al. (2018),
doi:10.1093/cercor/bhx179, and Yeo et al. (2011),
doi:10.1152/jn.00338.2011.

## Linear MNI152 template images

The three MNI152 2-mm images were copied from the installed FSL standard-data
distribution, as recorded in the input manifest and atlas notice. They use the
linear MNI152 space; they are not the supplied MNI2009c derivatives. MNI/ICBM
attribution is preserved. The MNI linear-template notice below is reproduced
from the original provider, checked 18 October 2026:
https://www.mcgill.ca/bic/software/tools-data-analysis/anatomical-mri/atlases/icbm152lin

> Copyright (C) 1993-2009 Louis Collins, McConnell Brain Imaging Centre,
> Montreal Neurological Institute, McGill University. Permission to use, copy,
> modify, and distribute this software and its documentation for any purpose
> and without fee is hereby granted, provided that the above copyright notice
> appear in all copies. The authors and McGill University make no representations
> about the suitability of this software for any purpose. It is provided "as is"
> without express or implied warranty. The authors are not responsible for any
> data loss, equipment damage, property loss, or injury to subjects or patients
> resulting from the use or misuse of this software package.

## External tools

FSL executables, libraries and installers are **not redistributed**. Raw imaging
recomputation needs the user's installed topup, eddy_cpu and FAST; its terms are
separate: https://fsl.fmrib.ox.ac.uk/fsl/docs/license.html . Version/build/hash
records in `config/fsl_environment.json` describe the original run, not binaries
installed by uv. Python dependencies retain their own licenses; uv.lock pins the
resolved packages. TeX and Poppler are external system dependencies.
