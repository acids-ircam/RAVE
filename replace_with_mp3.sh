for audio in $(ls *.wav)
do
    ffmpeg -y -loglevel panic -hide_banner -i $audio -b:a 128k -ac 1 $(basename $audio wav)mp3
    rm $audio
done