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
reader = easyocr.Reader(['th','en']) # this needs to run only once to load the model into memory

def clean_text(txt):
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
        
def clean_number(text):

    split_clean = [i.strip() for i in text] 

    clean_number = []
    for i in split_clean:
        cn = i
        txt = i.split(' ')
        for index,j in enumerate(txt):
            # if j and j+1 is number then combine it
            if index < len(txt) - 1 and re.match(r'[-0-9,.]*|[-0-9,.]*', j) and re.match(r'[-0-9,.]*|[-0-9,.]*', txt[index + 1]):
                cn = j + txt[index + 1]
                break
            elif re.match(r'[-0-9,.]*|[-0-9,.]*',j):
                cn = j
                break
        clean_number.append(cn)

    # if has o in front of number please change it to 0
    clean_number = [re.sub(r'[ocCDdq\[\}\(\)a๓กe]','0',i) for i in clean_number]
    #if has i or l or ! or | or I or f please change it to 1
    clean_number = [re.sub(r'[iI!l|fาเ?tว]','1',i) for i in clean_number]
    # if has s or S please change it to 5
    clean_number = [re.sub(r'[sS]','5',i) for i in clean_number]
    # if has g or q or Q or G please change it to 9
    clean_number = [re.sub(r'[gG?]','9',i) for i in clean_number]

    #   change , to .
    clean_number = [re.sub(r',|(\.\s)|(,\s)|(\s,\s)|(\s\.\s)','.',i) for i in clean_number]

    # if find more than 1 . please remov it except the last one
    clean_number = [re.sub(r'\.(?=.*\.)','',i) for i in clean_number]

    # if has - in front of number please remove it
    clean_number = [re.sub(r'.\*','',i) for i in clean_number]

    # if has * 
    clean_number = [re.sub(r'(.*-)|(\')','',i) for i in clean_number]

    # if has * 
    clean_number = [re.sub(r'\'','',i) for i in clean_number]

    # if has a char  
    clean_number = [re.sub(r'[a-zA-Zก-ฮ:\{\}]','',i) for i in clean_number]

    # if in clean_number =^\D. is True then put 00 behide it
    # clean_number = [i+'00' if re.match(r'(\d*\.(\W|))',i) else i for i in clean_number]
    # if not [0-9] OR dot then change it to 0
    clean_number = [re.sub(r'[^\d\.]','0',i) for i in clean_number]

    # if '' in clean_number then change it to 0
    clean_number = [re.sub(r'\'\'','0',i) for i in clean_number]

    #add string together
    clean_number = ''.join(clean_number)
    return clean_number

def clean_value(value):
    try:
        return float(value)
    except ValueError:
        return 0.0

def get_max_number(lst):
    max_num = 0.0
    for item in lst:
        if isinstance(item, str):
            # If the item is a mixture of text and number, try to convert to number
            num = clean_value(item)
            max_num = max(max_num, num)
        elif isinstance(item, (int, float)):
            # If the item is a number, compare with the current maximum
            max_num = max(max_num, item)
    return max_num

def parser(text,label):
    if label == 'ITEM':
        # text = text.lower()
        # text = re.sub(r'\D','',text)
        pass

    elif label in ('COST','TOTAL', 'VAT', 'SERVC', 'DISC'):
        text = clean_number(text)

    return text

# fuction format json
def format_json(datadict):
    items = []
    max_length = max(len(datadict['ITEM']), len(datadict['COST']))

    for i in range(max_length):
        item_name = datadict['ITEM'][i] if i < len(datadict['ITEM']) else ''
        cost_price = datadict['COST'][i] if i < len(datadict['COST']) else 0.0
        cost_price = clean_value(cost_price)
        items.append({'name': item_name, 'price': cost_price})

    # Clean the data
    cleaned_total = [clean_value(value) for value in datadict['TOTAL']]
    cleaned_vat = [clean_value(value) for value in datadict['VAT']]
    cleaned_servc = [clean_value(value) for value in datadict['SERVC']]
    cleaned_disc = [clean_value(value) for value in datadict['DISC']]
    # Get the maximum number
    total = get_max_number(cleaned_total)
    vat = get_max_number(cleaned_vat)
    servc = get_max_number(cleaned_servc)
    disc = get_max_number(cleaned_disc)

        # Constructing the new JSON structure
    new_json = {
        "item": items,
        "total": total,
        "vat": vat,
        "serviceCharge": servc,
        "discount": disc
    }
    return new_json


grp_gen = groupgen()

def get_predictions(image):
    tessData = reader.readtext(image, detail = 0)
    # convert into dataframe
    tessList = {'text' : tessData}
    df = pd.DataFrame(tessList)
    df.dropna(inplace=True) # drop missing values
    df['text'] = df['text'].apply(clean_text)

    # convet data into content
    df_clean = df.query('text != "" ')
    content = " ".join([w for w in df_clean['text']])
    print(content)
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

    datafram_tokens = datafram_tokens.query('token not in [".",","," "]')
    datafram_tokens['token_shift_-1'] = datafram_tokens['token'].shift(-1)
    datafram_tokens['token_shift_-2'] = datafram_tokens['token'].shift(-2)
    datafram_tokens['token_shift_1'] = datafram_tokens['token'].shift(1)
    datafram_tokens['token_shift_2'] = datafram_tokens['token'].shift(2)
    
    token_filter_cost = datafram_tokens['token'].fillna('').str.contains(r'(\d+[\,\.]\d+[\,\.]\d+)|(\d+[\,\.]\d+)|(^(\d)[\,\.]$\d)')
    token_filter_vat = datafram_tokens['token'].str.contains(r'^(tax|vat|ภาษี|t4x7)')
    token_filter_discount = datafram_tokens['token'].str.contains(r'^(ส่วนลด|discount|ลด)')
    token_filter_total = datafram_tokens['token'].str.contains(r'^(รวม|total|totaไ|grandtotal|crandtota|totaไ|tolal|tofal|รวมทั้งสิ้น|รวมสุทธิ|ฟั้ง|fatal|iitai|tota1|ยิดราม|ทังทมด|พัทมด)|สุทธํ$|สุทธิ$|ทังสิน$|รวม$|ราม$|มด$|tal$')
    token_filter_service = datafram_tokens['token'].str.contains(r'^(service|บริการ|ค่าบริการ|serice)|charge$|การ$|การ10$')
    # if token_shift_1 or	token_shift_2 ^(\u0E00-\u0E7Fa-zA-Z) and (datafram_tokens['label'] == 'O') = token_filter_item
    token_filter_item = datafram_tokens['token'].fillna('').str.contains(r'[\u0E00-\u0E7Fa-zA-Z]{5,}')
    # else condition 
    token_filter_else = datafram_tokens['token'].str.contains(r'^(เงิ็น|เง็น|เงิน|เง็นทอน|เงิ็นสด|beforevat|include)|included$|ทอน$|tem$')

    # Filter for 'token_shift_1' column containing a number with a decimal point, handling None values
    decimal_filter = datafram_tokens['token_shift_-1'].fillna('').str.contains(r'(\d+[\,\.]\d+[\,\.]\d+)|(\d+[\,\.]\d+)|(^(\d)[\,\.]$\d)')
    decimal_filter_spc = datafram_tokens['token_shift_-2'].fillna('').str.contains(r'(\d+[\,\.]\d+[\,\.]\d+)|(\d+[\,\.]\d+)|(^(\d)[\,\.]$\d)')

    # Combine the filters and update the 'label' column on the next row
    datafram_tokens.loc[token_filter_cost, 'label'] = 'B-COST'
    datafram_tokens.loc[(token_filter_total & (decimal_filter|decimal_filter_spc)).shift(fill_value=False), 'label'] = 'B-TOTAL'
    datafram_tokens.loc[(token_filter_vat & decimal_filter).shift(fill_value=False), 'label'] = 'B-VAT'
    datafram_tokens.loc[(token_filter_discount & decimal_filter).shift(fill_value=False), 'label'] = 'B-DISC'
    datafram_tokens.loc[(token_filter_service & decimal_filter).shift(fill_value=False), 'label'] = 'B-SERVC'
    datafram_tokens.loc[token_filter_item & (decimal_filter|decimal_filter_spc), 'label'] = 'B-ITEM'
    datafram_tokens.loc[token_filter_vat | token_filter_discount | token_filter_total | token_filter_service | token_filter_else, 'label'] = 'O'
    datafram_tokens.loc[(token_filter_else).shift(fill_value=False),'label'] = 'O'
    datafram_tokens.loc[(token_filter_cost & decimal_filter),'label'] = 'O'
    

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

    # format json
    entities = format_json(entities)
        
    return entities