#!/bin/bash
# compress mp4 videos

for file in *.mp4; do
	echo "compressing $file"
	output=compressed/"${file%.*}"_compressed.mp4
	ffmpeg -i "$file" -c:v libx264 -crf 23 -preset medium -c:a aac -b:a 128k -af "loudnorm=I=-16:TP=-1.5:LRA=11" "$output"
done
