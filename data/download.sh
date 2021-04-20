# Script for downloading the datasets 
# to this folder. 

# TODO
# 1. Upload the data to a static source 
#    Ex. Please research on https://git-lfs.github.com/ and see the pricing
#    Alternatively we can use a Dropbox link / GCP Bucket/ Amazon S3. 
#    Q: which one you think is better? 

wget -O "lfw.zip" "https://www.dropbox.com/sh/emevg60ys6wzny3/AAAsfskq3EusZpHjMBXtykzfa?dl=0";
sudo apt-get install unzip;
unzip "lfw.zip" -d "lfw";
rm "lfw.zip";


# 2. Use `wget` to download the data files to this folder from the links above.
#    Additionally, add the files to the gitignore, so this repo does not track. 
# 3. Adjust the configuration in the `source` folder. 
