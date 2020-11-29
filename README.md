Folder structure

```
golden-retriever
|_ data                            #folder to store processed (augmented) malay constituent data
|_ EVALB                           #folder with program to run evaluation on constituency results
|_ export_model                    #folder to store converted tf graph model from pytorch
|_ models                          #folder to store saved pytorch .pt models during training
|_ output                          #folder to store test, prediction and evaluation score for val and test set
|_ preprocess                      #folder with preprocess file to augment indonesian dataset with malay synonyms
|_ src                          
    |_ static                      #folder with css, js, etc. scripts for interface
    |_ templates                   #folder with html interface
    |_ chart_decoder_py.pyx        #constituent chart decoder script used for interface inferencing
    |_ chart_helper.pyx            #constituent chart helper script used during training
    |_ evaluate.py                 #evaluation script used during training  
    |_ export.py                   #export script to convert pytorch .pt model into tf graph model, with quantized version available
    |_ inference.py                #inference script for fast.api interface on model predictions
    |_ main_xlnet_base.py          #main training script for xlnet base model
    |_ nkutil.py                   #script for helper function for parameters
    |_ parse_nk_xlnet_base.py      #script for model, batch and tree parsing functions
    |_ tokens.py                   #addition tokens used for model
    |_ trees_newline.py            #functions for tree parsing
    |_ vocabluary.py               #vocabulary storing function
|_ requirements.txt
```

References: <br>
https://github.com/huseinzol05/Malaya/tree/master/session/constituency<br>
https://github.com/nikitakit/self-attentive-parser

Create and activate environment 

`python3 -m venv malaynlp` <br>
`source {environment filepath}/bin/activate`

Install dependencies

`pip install -r requirements.txt`

1. Run preprocessing

`python3 preprocess/preprocess.py`

2. Run training, where model-path-base is filepath to save model

`python3 src/main_xlnet_base.py train --use-bert --model-path-base models/xlnet --num-layers 2 --learning-rate 0.00005 --batch-size 32 --eval-batch-size 16 --subbatch-max-tokens 500 --predict-tags --train-path data/train-aug.txt --dev-path data/val-aug.txt`

3. Export pytorch model into tf graph, where model_path is where pytorch pt model was saved

`python3 src/export.py --model_path models/xlnet_dev=82.50.pt --test_path data/test-aug.txt --export_path export_model/`

4. Run inference interface

`cd src`
`uvicorn inference:app`

Data for training-3502, val-448, test-454.<br>
Current F1 score on test set is 79.35%.