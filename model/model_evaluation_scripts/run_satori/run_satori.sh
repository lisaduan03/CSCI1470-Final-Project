#!/bin/sh
#SBATCH -J satori
#SBATCH -o /users/jcurrie2/data_koconno5/jcurrie2/satori/no_rnn_filters_100_fsize_20_highthresh/satori_FLYFACTOR_highthresh.out
#SBATCH -e /users/jcurrie2/data_koconno5/jcurrie2/satori/no_rnn_filters_100_fsize_20_highthreshm/satori_FLYFACTOR_highthresh.err
#SBATCH -p gpu --gres=gpu:1
#SBATCH -n 2
#SBATCH -t 1:00:00
#SBATCH --mem 100G
#SBATCH --mail-type=END
#SBATCH --mail-user=justin_currie@brown.edu

module load miniconda3
source activate /users/jcurrie2/.conda/envs/satori

python3 satori.py /users/jcurrie2/data_koconno5/jcurrie2/embryo_new_annotations/12_16/12_16_synaptic_randomA_enhancers_stranded hparamfile.txt \
        --verbose \
        --outDir /users/jcurrie2/data_koconno5/jcurrie2/satori/no_rnn_filters_100_fsize_20_highthresh \
        --mode train \
        --splitperc 20 \
        --motifanalysis \
        --scorecutoff 0.75 \
        --tomtompath /users/jcurrie2/data_koconno5/jcurrie2/MEME/meme-5.5.5/src/tomtom \
        --database /users/jcurrie2/data_koconno5/jcurrie2/MEME/fly_factor_survey_with_clamp.meme \
        --annotate /users/jcurrie2/run_satori/TF_Information_all_motifs_plus.txt \
        --interactions \
        --background negative \
        --attncutoff 0.04 \
        --fiscutoff 0 \
        --numlabels 2 \
        --tomtomdist ed \
        --tomtompval 0.05 \
        --testall \
        --attrbatchsize 12 \
        --method SATORI
        

        
        
