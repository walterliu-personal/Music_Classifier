# Music_Classifier
Part of the source code for classifying music. Graduation project for the EDT Insight into University Course 2022

Usage:
git clone https://github.com/WalterLiu417/Music_Classifier.git
pip install -r requirements.txt

Make a copy of this Colab notebook by Roboflow: https://colab.research.google.com/drive/1Z1lbR_oTSaeodv9tTm11uEhOjhkUx1L4?usp=sharing#scrollTo=5ql2T5PDUI1D
Upload dataset.zip to the Colab runtime and unzip
Train the model with epochs=1 and batch_size=50
Save the model

Read MusicXML files with music21.corpus.parse(FILE_PATH)
Use the main function in score_to_img.py to convert the score object to an image
Use the infer function in infer.py to test the model with that image

Thanks to music21 and Roboflow
