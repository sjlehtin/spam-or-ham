#!/bin/sh

CMD="./decisiontree.py --dump-tree original --dump-tree pruned"
OUT="data.out"
echo "Starting gathering..." > $OUT
for ds in 1000 2000 4000 6000 8000; do
    echo "" >> $OUT
    echo "**** $ds, postpruning only" >> $OUT
    CUR=out-$ds-$$.txt
    (time $CMD --training-data $ds) >& $CUR
    tail -20 $CUR >> $OUT

    cp classified-original.dot postpruned-original-$ds.dot
    cp classified-pruned.dot postpruned-pruned-$ds.dot

    echo "" >> $OUT
    echo "**** $ds, pre- and postpruned" >> $OUT
    CUR=out-$ds-$$.txt
    (time $CMD --preprune --training-data $ds) >& $CUR
    tail -20 $CUR >> $OUT

    cp classified-original.dot prune-both-original-$ds.dot
    cp classified-pruned.dot prune-both-pruned-$ds.dot

done
