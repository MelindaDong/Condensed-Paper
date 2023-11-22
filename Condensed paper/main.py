# set global variables
paper_path = 'testing_paper/RWKV_25.pdf'


# import libraries
from functions import get_ordered_layout_list
import functions as F
import gpt_function
import layoutparser as lp
import numpy as np
import re
from PIL import Image
import pandas as pd


# set GPT API key
from dotenv import load_dotenv
import os
# Load environment variables from .env file
load_dotenv()
api_key = os.environ.get("OPENAI_API_KEY")


## PART 1: LAYOUT PARSER: PARSING PAPER
# load model
model = lp.Detectron2LayoutModel('lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config', 
                                 extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.8],
                                 label_map={0: "Text", 1: "Title", 2: "List", 3:"Table", 4:"Figure"})
# load paper
layout_list, image_list = get_ordered_layout_list(model, paper_path)

# get title list
title_text = ""
for page in layout_list:
    for element in page:
        if element.type == 'Title':
            title_text += element.text 

# save image and {image_path:caption} dictionary
image_discription_dict = {}

for i, page in enumerate(layout_list):
    for j, element in enumerate(page):
        # get rid of the reference part
        if element.type == 'Text' and element.text == 'References\n':
            break
        if element.type == 'Text' or element.type == 'Title' or element.type == 'List': # List会包括进很多references
            continue
        else:
            # save the element as a .png file
            file_name = f"other_images/{element.type}_{i}_{j}.png"
            image_array = element.crop_image(image_list[i])

            # adjust the size of the image
            image = Image.fromarray(image_array)
            image.save(file_name)

            image_discription = F.get_image_discription(page, j)
            image_discription_dict[file_name] = image_discription

# convert the image_discription_dict to a dataframe
image_discription_df = pd.DataFrame.from_dict(image_discription_dict, orient='index')
image_discription_df.reset_index(inplace=True)
image_discription_df.columns = ['image', 'discription']

def get_the_key(discription):
    return discription[:9] #数出来的是9个字符

# extract the key words from the image discription text and save it as new column
image_discription_df['key_words'] = image_discription_df['discription'].apply(get_the_key)

def find_keys(raw_text, key):
    if key == "":
        return ""
    keys = [key]
    if key[0] == "F":
        keys = [key[:3] + '. ' + key[-2:-1], key[:-1], key] # ['Fig. 1', 'Figure 1']
    if key[0] == "T":
        keys = [key[:-2], key[:-1]] # ['Table 1', 'Table 1.']
    return keys

### use pyPDF2 extract text from pdf
raw_text = F.extract_raw_text_from_pdf(paper_path)

if "References" in raw_text:
    raw_text = re.sub(r'References.*$', '', raw_text)

# replace the \n with space
raw_text = raw_text.replace('\n', ' ')

# apply find_contexts function to the key_words column and save it as new column
image_discription_df['keys'] = image_discription_df.apply(lambda x: find_keys(raw_text, x['key_words']), axis=1)

# save the image_discription_df as csv file
image_discription_df.to_csv('image_discription_df.csv', index=False)

### get the method_sumamry
# turn title_text into a list
title_list = title_text.split('\n')

# get the summary 2: just method
method_index = gpt_function.get_method_title(title_list) #'4(3. BERT):7(4 Experiments)'

def extract_text_between(raw_text, start_session, end_session):
    start_index = raw_text.find(start_session)
    end_index = raw_text.find(end_session)

    if start_index != -1 and end_index != -1:
        extracted_text = raw_text[start_index + len(start_session):end_index]
        return extracted_text
    else:
        # remove "." in the start_session and end_session
        start_session = start_session.replace(".", "")
        end_session = end_session.replace(".", "")
        start_index = raw_text.find(start_session)
        end_index = raw_text.find(end_session)
        extracted_text = raw_text[start_index + len(start_session):end_index]
        return extracted_text
    
# start title : end title
start_index0 = method_index.split(":")[0]
# remove the braket and the text inside from the start_index0
start_index = start_index0.split("(")[0]

end_index0 = method_index.split(":")[1]
end_index = end_index0.split("(")[0]

start_session = title_list[int(start_index)]
end_session = title_list[int(end_index)]

method_text = extract_text_between(raw_text, start_session, end_session)

# add all the keys to a list
total_keys = []
for keys in image_discription_df['keys']:
    total_keys += keys

# keys in method_text: find the keys in the method_text
method_keys = []
for key in total_keys:
    if key in method_text:
        method_keys.append(key)

### get the method_summary
# if method_text is less than 6000, directly use get_method_summary
if len(method_text.split(" ")) < 6000:
    method_summary = gpt_function.get_method_summary(method_text, method_keys)

# if method_text is more than 6000, split it into two parts
else:
    method_text2 = method_text.split(" ")[:6000]
    method_summary = gpt_function.get_method_summary(method_text, method_keys)

method_summary = "### Detailed Method:\n\n"  + method_summary

# save method_summary to markdown file
with open("Method_summary.md", "w") as f:
    f.write(method_summary)

### get the overview summary
_, summary, filtered_QA_pair = gpt_function.generate_summary(raw_text, api_key)

### combine summary and method_summary to V1_summary
V1_summary = summary + "\n\n" + method_summary

## add the title of the paper 
paper_title = ""
for item in title_list:
    # if the first letter of the item is a number, break
    if item[0].isnumeric():
        break
    else:
        paper_title += str(item)

paper_title = "# " + paper_title + "\n\n"
V1_summary0 = paper_title + V1_summary

# save V1_summary to txt
with open("V1_summary.md", "w") as f:
    f.write(V1_summary0)

print("get V1_summary successfully!")
#------------------------------------------------------------
has_equation, math_dict = F.has_equation(paper_path)
print("check math successfully! has equation: ", has_equation)
#------------------------------------------------------------

# read V1_summary as text
with open ('V1_summary.md', 'r') as f:
    V1_summary = f.read()
V1_summary = V1_summary + '\n'

# read math_ocr 
with open ('math_ocr_text.txt', 'r') as f:
    math_ocr_text = f.read()
# convert imgae_discription_text to dictionary
math_ocr_dict = eval(math_ocr_text)

from langchain.document_loaders import TextLoader
loader = TextLoader('Method_summary.md')

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings

if math_ocr_dict != {}:
    
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        # Set a really small chunk size, just to show.
        chunk_size = 80,
        chunk_overlap  = 20,
        length_function = len,
        is_separator_regex = False,
    )
    texts = text_splitter.split_documents(documents)
    embeddings = OpenAIEmbeddings()
    db = FAISS.from_documents(texts, embeddings)
    retriever = db.as_retriever(search_type="similarity_score_threshold", search_kwargs={"score_threshold": .9})

    # read the query from image_discription_dict value
    query_list = list(math_ocr_dict.values()) 

    doc_list = []
    for query in query_list:
        doc = retriever.get_relevant_documents(query)
        if len(doc) == 0:
            doc_list.append('NA')
        else:
            doc_list.append(doc[0].page_content)

    # get the non-NA doc_list
    target_doc = [doc for doc in doc_list if doc != 'NA']

    target_index = [i for i, doc in enumerate(doc_list) if doc != 'NA']
    # Get the keys based on the index list
    keys_from_index = [list(math_ocr_dict.keys())[index] for index in target_index]

    for i in range(len(target_doc)):
        target_key = keys_from_index[i]
        pic_adding = f'<br><img src="{target_key}" alt="alt text" style="max-width: 60%; height: auto;"><br>'
        
        doc = target_doc[i]
        end_index = V1_summary.find(doc) + len(doc)
        # make sure pic adding is not in the middle of a word
        while V1_summary[end_index] != '\n':
            end_index += 1

        V1_summary = V1_summary[:end_index]  + pic_adding  + V1_summary[end_index:]

else:
    print('No math detected')

# save the test_V1_summary to V1.5_summary.md(add math images)
with open('V1.5_summary.md', 'w') as f:
    f.write(V1_summary)

print("get V1.5_summary successfully!")
#------------------------------------------------------------
print("method_keys: ", method_keys)

# read V1.5_summary as text, in case 1.5 version added math images
with open ('V1.5_summary.md', 'r') as f:
    V1_summary = f.read()

# remove the row if there is NaN
image_discription_df = image_discription_df.dropna()
# remove the rows with repeated discription
image_discription_df = image_discription_df.drop_duplicates(subset='discription', keep='first')

if method_keys != []:

    # Filter the DataFrame
    filtered_df = image_discription_df[image_discription_df['keys'].apply(lambda x: any(key in x for key in method_keys))]
    filtered_df = filtered_df.reset_index(drop=True)

    # use re to get the text after '### Detailed Method' otherwise(all pics matches the overall part)
    method_summary = re.split('### Detailed Method:', V1_summary)[1]
    # turn method_summary into a list of sentences
    method_summary_list = method_summary.split('.')

    def get_the_i(item, filtered_df):
        # if the item is in the column 'keys' of filtered_df, return the index of that row
        for i in range(len(filtered_df)):
            if item in filtered_df.iloc[i]['keys']:
                return i

    # for each item in method_keys, use RE find it in method_summary
    # if found, print the item and the sentence that contains it
    used_indices = set()
    for item in method_keys:
        for index, sentence in enumerate(method_summary_list):
            if re.search(item, sentence):
            
                i = get_the_i(item, filtered_df)
                # Check if this index has already been used
                if i not in used_indices:
                    used_indices.add(i)

                    target_key = filtered_df['image'][i]
                    target_caption = filtered_df['discription'][i]
                    pic_adding = f'<br><img src="{target_key}" alt="alt text" style="max-width: 80%; height: auto;"><br><sub style="color: gray;">{target_caption}</sub><br>'


                    sentence = sentence + ". " + pic_adding
                    method_summary_list[index] = sentence
    
    # concatenate the sentences in method_summary_list into a string, add "." between each sentence
    new_method_summary = ".".join(method_summary_list)

    # add title and overview summary and the new_method_summary together
    V2_summary = paper_title + summary + "\n\n" + "### Detailed Method:\n\n" + new_method_summary

    # save V2_summary to markdown file
    with open("V2_summary.md", "w") as f:
        f.write(V2_summary)

    print("get V2_summary successfully!")
else:
    print("There's no in-text reference in the method part, so no V2_summary generated!")
