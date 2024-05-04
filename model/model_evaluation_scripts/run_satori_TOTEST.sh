#!/bin/sh
#SBATCH -J satori
#SBATCH -o /users/jcurrie2/data_koconno5/jcurrie2/satori/simulated_results/slightly_connected/satori_FLYFACTOR.out
#SBATCH -e /users/jcurrie2/data_koconno5/jcurrie2/satori/simulated_results/slightly_connected/satori_FLYFACTOR.err
# #SBATCH -p gpu --gres=gpu:1
#SBATCH -n 15
#SBATCH -t 00:10:00
#SBATCH --mem 30G
#SBATCH --mail-type=END
#SBATCH --mail-user=justin_currie@brown.edu

module load miniconda3
source activate /users/jcurrie2/.conda/envs/satori_new

python3 satori_mine.py /users/jcurrie2/promoter_predict/250_50_dros_promoters_background_epd hparamfile.txt \
        --verbose \
        --outDir /users/jcurrie2/data_koconno5/jcurrie2/satori/simulated_results/slightly_connected \
        --mode test \
        --splitperc 10 \
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
        

        
        
