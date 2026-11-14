# Data and Code Availability

The LGCS demonstration code, configuration files, synthetic example data, data-generation script, and tests are publicly available at [https://github.com/liyanan010401-wq/LGCS-demo](https://github.com/liyanan010401-wq/LGCS-demo). The release demonstrates the graph-to-table interface and context-selection workflow using a small graph encoder and a surrogate context predictor; it does not include the complete real-data experimental implementation. The bundled data are entirely synthetic and contain no patient information. EyePACS and RFMiD are not redistributed; access to these datasets should be obtained from their original providers under the applicable terms. Synthetic demonstration outputs must not be interpreted as experimental or clinical validation of the proposed method.

## Original dataset sources

- **EyePACS:** [Diabetic Retinopathy Detection competition data on Kaggle](https://www.kaggle.com/competitions/diabetic-retinopathy-detection/data).
- **RFMiD:** [original dataset publication](https://doi.org/10.3390/data6020014) and its linked [all-classes dataset page](https://riadd.grand-challenge.org/download-all-classes/).

These are provider references for the manuscript's proposed real-data protocol. This demo has not run an EyePACS/RFMiD evaluation and does not assert that those datasets generated any of its outputs. The synthetic array designation documented in `data/DATA_CARD.md` does not change the terms of external datasets or software. No official SIGraph or TabICL checkpoint is distributed.
