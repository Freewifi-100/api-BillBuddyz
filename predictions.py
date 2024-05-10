import numpy as np
import pandas as pd
import cv2
import easyocr
from glob import glob
import spacy
import re
import string
import warnings
warnings.filterwarnings('ignore')

### Load NER model
model_ner = spacy.load('./model/model-best_v1/')
reader = easyocr.Reader(['th','en'], gpu=False) # this needs to run only once to load the model into memory

def cleanText(txt):
    whitespace = string.whitespace
    punctuation = "!#$%&\'()*+:;<=>?[\\]^`{|}~"
    tableWhitespace = str.maketrans('','',whitespace)
    tablePunctuation = str.maketrans('','',punctuation)
    text = str(txt)
    #text = text.lower()
    removewhitespace = text.translate(tableWhitespace)
    removepunctuation = removewhitespace.translate(tablePunctuation)
    
    return str(removepunctuation)

# group the label
class groupgen():
    def __init__(self):
        self.id = 0
        self.text = ''
        
    def getgroup(self,text):
        if self.text == text:
            return self.id
        else:
            self.id +=1
            self.text = text
            return self.id
        


def parser(text,label):
    if label == 'ITEM':
        # text = text.lower()
        # text = re.sub(r'\D','',text)
        pass

    elif label in ('COST','TOTAL'):
        text = text.lower()
        allow_special_char = '.,'
        text = re.sub(r'[^A-Za-z0-9{} ]'.format(allow_special_char),'',text)
        text = text.replace(',','.')

    elif label in ('VAT', 'SERVC'):
        text = text.lower()
        text = re.sub(r'[^a-z ]','',text)
        text = text.title()

    elif label == 'DISC':
        text = text.lower()
        text = re.sub(r'[^a-z0-9 ]','',text)
        text = text.title()

    return text


grp_gen = groupgen()

def getPredictions(image):
    tessData = reader.readtext(image, detail = 0)
    # convert into dataframe
    tessList = {'text' : tessData}
    df = pd.DataFrame(tessList)
    df.dropna(inplace=True) # drop missing values
    df['text'] = df['text'].apply(cleanText)

    # convet data into content
    df_clean = df.query('text != "" ')
    content = " ".join([w for w in df_clean['text']])
    # print(content)
    # get prediction from NER model
    doc = model_ner(content)

    # converting doc in json
    docjson = doc.to_json()
    doc_text = docjson['text']

    # creating tokens
    datafram_tokens = pd.DataFrame(docjson['tokens'])
    datafram_tokens['token'] = datafram_tokens[['start','end']].apply(
        lambda x:doc_text[x[0]:x[1]] , axis = 1)

    right_table = pd.DataFrame(docjson['ents'])[['start','label']]
    datafram_tokens = pd.merge(datafram_tokens,right_table,how='left',on='start')
    datafram_tokens.fillna('O',inplace=True)

    # join lable to df_clean dataframe
    df_clean['end'] = df_clean['text'].apply(lambda x: len(x)+1).cumsum() - 1 
    df_clean['start'] = df_clean[['text','end']].apply(lambda x: x[1] - len(x[0]),axis=1)

    # inner join with start 
    dataframe_info = pd.merge(df_clean,datafram_tokens[['start','token','label']],how='inner',on='start')

    # Bounding Box

    bb_df = dataframe_info.query("label != 'O' ")

    bb_df['label'] = bb_df['label'].apply(lambda x: x[2:])
    bb_df['group'] = bb_df['label'].apply(grp_gen.getgroup)

    # Entities

    info_array = dataframe_info[['token','label']].values
    entities = dict(ITEM=[],COST=[],TOTAL=[],VAT=[],SERVC=[],DISC=[])
    previous = 'O'

    for token, label in info_array:
        bio_tag = label[0]
        label_tag = label[2:]

        # step -1 parse the token
        text = parser(token,label_tag)

        if bio_tag in ('B','I'):

            if previous != label_tag:
                entities[label_tag].append(text)

            else:
                if bio_tag == "B":
                    entities[label_tag].append(text)

                else:
                    if label_tag in ("ITEM",'COST','DISC','VAT','TOTAL','SERVC'):
                        entities[label_tag][-1] = entities[label_tag][-1] + " " + text

                    else:
                        entities[label_tag][-1] = entities[label_tag][-1] + text

        previous = label_tag
        
    return entities
