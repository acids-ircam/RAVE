function cleanup(){
    rm job_*.sh
    rm jobs.input
}


cleanup

INDEX=0
for file in $(ls instruction_*.txt)
do
    cat $file | grep python > job_$INDEX.sh
    echo job_$INDEX.sh >> jobs.input
    ((INDEX++))
done

cat jobs.input | parallel -j$1 'CUDA_VISIBLE_DEVICES=$(({%}-1)) source {} > job_$(({#}-1)).log 2>&1'

cleanup