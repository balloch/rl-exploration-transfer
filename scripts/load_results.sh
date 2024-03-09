if [[ $# -eq 0 ]] ; then
    echo 'Please provide a name to save these results under.'
    exit 1
fi

rm -rf figures/*
rm -rf data/*

mv results/$1/figures/* figures
mv results/$1/data/* data
rm -rf results/$1

