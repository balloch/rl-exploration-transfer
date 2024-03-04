if [[ $# -eq 0 ]] ; then
    echo 'Please provide a name to save these results under.'
    exit 1
fi

mkdir -p results/$1
rm -rf results/$1
mkdir -p results/$1/figures
mkdir -p results/$1/data
mv figures/* results/$1/figures
mv data/* results/$1/data
