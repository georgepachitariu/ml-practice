"""
Implementations of Faster R-CNN models must define a new
FasterRCNNFeatureExtractor and override three methods: `preprocess`,
`_extract_proposal_features` (the first stage of the model), and
`_extract_box_classifier_features` (the second stage of the model). Optionally,
the `restore_fn` method can be overridden.  See tests for an example.

To use InceptionV100, we will have to define a new FasterRCNNFeatureExtractor and pass it
to our FasterRCNNMetaArch constructor as input.
See object_detection/meta_architectures/faster_rcnn_meta_arch.py
for definitions of FasterRCNNFeatureExtractor and FasterRCNNMetaArch, respectively.
A FasterRCNNFeatureExtractor must define a few functions:
"""

import tensorflow
import faster_rcnn_meta_arch as arch


x = arch.FasterRCNNMetaArch()

if __name__ == "__main__":
 print('Hi')