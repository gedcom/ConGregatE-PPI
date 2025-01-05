for seed in {1..10}; do
	python ConGreGatE-PPI_code/ppipredict.py --seed $seed --testset dev
done
