For the convenience of phased debugging, the Python source code of the project has not been integrated. Taking the "spoof_the_drive_gear" attack as an example, run the following programs in sequence to achieve data learning and generation:

1. Run "DataPreprocessing.py" in the current directory to extract the raw data of the CAN data frame and perform simple processing.

2. Run "DataConversion.py" in the current directory to convert the raw information extracted in step 1 into binary format.

3. Run "generation_SpoofGear.py" in the current directory to learn the data probability distribution of the "spoof_the_drive_gear" type of cyber attack and generate new attack samples.

4. Run "CosSimilarity.py" in the current directory to calculate the cosine similarity between the generated data and the original data.

5. Run "BinaryToDecimalConversion" in the current directory to convert the generated CAN data in binary format into decimal representation. 