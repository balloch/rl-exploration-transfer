# python experiments/plot_data.py -pfw t
# python experiments/plot_data.py -pfw f --algs rnd noisy ngu rise diayn none
# python experiments/plot_data.py -pfw f --algs rnd icm girm ride none
python experiments/calculate_metrics.py -pfw t
./scripts/save_results.sh $1
