#!/bin/bash
x=1
for i in $(ls | grep .jpg)
do
	if [[ "${i#*.}" = jpg ]]
	then
		echo "${i#*.}"
	#statements
		##echo $i
		##echo $x
		if [[ $x -le 9 ]]
		then
			name=image_000$x
		elif [[ $x -le 99 ]] 
		then
			name=image_00$x
		elif [[ $x -le 999 ]]
		then
			name=image_0$x
		else
			name=image_$x
		fi
		echo $name.jpg
		mv $i $name.jpg
		cp $name.jpg rot1_$name.jpg
		mogrify -rotate 90 rot1_$name.jpg
		cp rot1_$name.jpg ups_$name.jpg
		mogrify -rotate 90 ups_$name.jpg
		cp ups_$name.jpg rot2_$name.jpg
		mogrify -rotate 90 rot2_$name.jpg
		(( x += 1 ))
		##echo $x
	fi
done

echo $( ls -1q *jpg | wc -l )
