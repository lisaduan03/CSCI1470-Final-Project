# CSCI1470-Final-Project

In this project, we reimplemented SATORI (Self-ATtentiOn for Regulatory 
Interactions) in tensorflow, augmented its architecture in minor ways to 
improve performance and interpretability, and tested it on a new dataset of
*D. melanogaster* promoters. See our full paper for more details on our process
and experimentation along the way.

## Code Structure

In the data folder, we have all data for 5001 promoters from the eukaryotic
promoter database for *D. melanogaster*. We preprocess and utilize this data
in the model directory.

In the model folder, we have our model (in model.py), our simulator for
generating clean motif interaction data (in simulator.py), our results on real
and simulated data (in real and simulated results respectively), and our scripts
for attempting to pull self-attention weights out of our model in a biologically
meaningful way (model_evaluation_scripts).

## Known Bugs

We have no know bugs in our approach, although our hyperparameter choices are
arbitrary beyond the experiments we ran (i.e. for number of channels, number
of attention heads). Note that the width of our filter in the 1D convolution
should be the expected length of a motif in your dataset.

## Miscellaneous Notes

Our model was successful at pulling out real motifs from real-world and
simulated data, in addition to achieving high accuracy on our promoter dataset.
However, we were unable to get conclusive results for pulling out motif
interactions from the self-attention weights. Our efforts in this direction are
in the model_evaluation_scripts directory, which we recycled and modified for
our purposes from the original SATORI publication. We also used these scripts to
generate sequence logos for our convolutional filters. Note that while we did
reuse much of this code from the original paper, it is not our model (which we 
programmed on our own in tensorflow), nor did it provide any tangible results to 
our actual paper (we were unable to replicate self-attention results) – we just 
wanted to document the effort.

Another way to look at motif interactions is by passing in a sequence that
contains two motifs that you hypothesize might interact to a trained model, and
seeing whether the max pools these motifs contribute to have high attention 
scores in the attention step of the model. We had some success with this 
approach on a small scale, and found that our model can learn motif interactions
(i.e., has high attention between interacting motifs on simulated data).

## Reading More

These are just some broad notes on our reimplementation of SATORI, but please
read our paper for more detail!
