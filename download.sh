# Download the datasets and checkpoints

echo 'Downloading the YCB-Video Dataset'

wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1if4VoEXNx9W3XCn0Y7Fp15B4GpcYbyYi' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1if4VoEXNx9W3XCn0Y7Fp15B4GpcYbyYi" -O YCB_Video_Dataset.zip && rm -rf /tmp/cookies.txt
unzip YCB_Video_Dataset.zip
mv YCB_Video_Dataset/ datasets/ycb/
rm YCB_Video_Dataset.zip

echo 'Downloading the preprocessed LineMOD dataset'

wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1YFUra533pxS_IHsb9tB87lLoxbcHYXt8' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1YFUra533pxS_IHsb9tB87lLoxbcHYXt8" -O Linemod_preprocessed.zip && rm -rf /tmp/cookies.txt
unzip Linemod_preprocessed.zip
mv Linemod_preprocessed/ datasets/linemod/
rm Linemod_preprocessed.zip

echo 'Downloading the trained checkpoints...'

wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1bQ9H-fyZplQoNt1qRwdIUX5_3_1pj6US' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1bQ9H-fyZplQoNt1qRwdIUX5_3_1pj6US" -O trained_checkpoints.zip && rm -rf /tmp/cookies.txt
unzip trained_checkpoints.zip
echo 'Copying trained checkpoints to trained_models/'
cp trained_checkpoints/ycb/*.pth trained_models/ycb/
cp trained_checkpoints/linemod/*.pth trained_models/linemod/
rm trained_checkpoints.zip
rm -r trained_checkpoints/

echo 'done'