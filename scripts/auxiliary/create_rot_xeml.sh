for i in $(ls | grep .xml)
do
	##i="${i%.*}"
	cp $i rot1_$i
	cp $i ups_$i
	cp $i rot2_$i
done